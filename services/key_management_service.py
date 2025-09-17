"""
Encryption Key Management Service
Handles generation, storage, rotation, and lifecycle management of encryption keys
"""

import os
import hmac
import hashlib
import secrets
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import base64

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidKey

from shared.database import db_manager
from shared.models import EncryptionKey, KeyUsageLog, KeyRotationHistory, KeyAccessAudit
from shared.events import EventBus, EventChannel

logger = logging.getLogger(__name__)


@dataclass
class KeyMetadata:
    """Metadata for an encryption key"""
    key_id: str
    key_type: str
    name: str
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime]
    rotation_count: int
    security_level: str


@dataclass
class KeyRotationResult:
    """Result of a key rotation operation"""
    success: bool
    old_key_id: str
    new_key_id: str
    rotation_time_ms: int
    error_message: Optional[str] = None


class KeyManagementService:
    """Service for managing encryption keys and their lifecycle"""

    def __init__(self):
        """Initialize the key management service"""
        self.backend = default_backend()
        self.event_bus = EventBus("key_management")
        self.event_bus.start()

        # Master key configuration
        self.master_key_id = os.getenv("ENCRYPTION_MASTER_KEY_ID", "master_default")
        self.master_key_passphrase = os.getenv("ENCRYPTION_MASTER_PASSPHRASE")

        if not self.master_key_passphrase:
            logger.warning("ENCRYPTION_MASTER_PASSPHRASE not set - using default (not secure for production)")
            self.master_key_passphrase = "default_master_passphrase_change_in_production"

        logger.info("KeyManagementService initialized")

    def generate_master_key(self) -> str:
        """Generate a new master encryption key"""
        try:
            # Generate a 256-bit (32-byte) master key
            master_key = secrets.token_bytes(32)

            # Encrypt the master key with the passphrase for storage
            encrypted_master_key = self._encrypt_with_passphrase(master_key, self.master_key_passphrase)

            # Store the encrypted master key
            with db_manager.get_db() as session:
                # Check if master key already exists
                existing = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == self.master_key_id,
                    EncryptionKey.key_type == 'master'
                ).first()

                if existing:
                    # Update existing master key
                    existing.encrypted_key = encrypted_master_key
                    existing.key_hash = self._calculate_key_hash(master_key)
                    existing.updated_at = datetime.utcnow()
                    existing.rotation_count += 1
                    session.commit()
                    logger.info(f"Updated existing master key: {self.master_key_id}")
                else:
                    # Create new master key
                    master_key_record = EncryptionKey(
                        key_id=self.master_key_id,
                        key_type='master',
                        name='Master Encryption Key',
                        description='Primary master key for encrypting data encryption keys',
                        encrypted_key=encrypted_master_key,
                        key_hash=self._calculate_key_hash(master_key),
                        algorithm='aes-256-gcm',
                        status='active',
                        security_level='critical',
                        expires_at=None,  # Master key doesn't expire
                        created_by='system'
                    )
                    session.add(master_key_record)
                    session.commit()
                    logger.info(f"Created new master key: {self.master_key_id}")

            # Audit the key generation
            self._audit_key_access(self.master_key_id, 'store', 'system', 'Master key generated/updated')

            return self.master_key_id

        except Exception as e:
            logger.error(f"Error generating master key: {e}")
            raise

    def get_master_key(self) -> bytes:
        """Retrieve and decrypt the master key"""
        try:
            with db_manager.get_db() as session:
                master_key_record = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == self.master_key_id,
                    EncryptionKey.key_type == 'master',
                    EncryptionKey.status == 'active'
                ).first()

                if not master_key_record:
                    raise ValueError(f"Master key not found: {self.master_key_id}")

                # Decrypt the master key
                master_key = self._decrypt_with_passphrase(
                    master_key_record.encrypted_key,
                    self.master_key_passphrase
                )

                # Verify integrity
                if self._calculate_key_hash(master_key) != master_key_record.key_hash:
                    raise ValueError("Master key integrity check failed")

                # Audit the access
                self._audit_key_access(self.master_key_id, 'retrieve', 'system', 'Master key retrieved')

                return master_key

        except Exception as e:
            logger.error(f"Error retrieving master key: {e}")
            raise

    def generate_data_key(self, table_name: str, key_name: Optional[str] = None) -> str:
        """Generate a data encryption key for a specific table"""
        try:
            master_key = self.get_master_key()

            # Generate a unique key ID
            key_id = f"data_{table_name}_{secrets.token_hex(8)}"
            if key_name:
                key_id = f"data_{table_name}_{key_name}"

            # Derive the data key from master key using HKDF
            data_key = self._derive_key(master_key, f"data_key_{table_name}", 32)

            # Encrypt the data key with master key
            encrypted_data_key = self._encrypt_data(data_key, master_key)

            # Calculate expiration (90 days from now)
            expires_at = datetime.utcnow() + timedelta(days=90)
            next_rotation_at = datetime.utcnow() + timedelta(days=90)

            # Store the encrypted data key
            with db_manager.get_db() as session:
                data_key_record = EncryptionKey(
                    key_id=key_id,
                    key_type='data',
                    name=f"Data Key for {table_name}",
                    description=f"Data encryption key for table: {table_name}",
                    encrypted_key=encrypted_data_key,
                    key_hash=self._calculate_key_hash(data_key),
                    algorithm='aes-256-gcm',
                    parent_key_id=self._get_master_key_record_id(),
                    derived_from=f"master_key:{self.master_key_id}",
                    status='active',
                    security_level='high',
                    expires_at=expires_at,
                    next_rotation_at=next_rotation_at,
                    created_by='system'
                )
                session.add(data_key_record)
                session.commit()

            # Audit the key generation
            self._audit_key_access(key_id, 'store', 'system', f'Data key generated for table: {table_name}')

            logger.info(f"Generated data key: {key_id} for table: {table_name}")
            return key_id

        except Exception as e:
            logger.error(f"Error generating data key for table {table_name}: {e}")
            raise

    def generate_field_key(self, table_key_id: str, field_name: str) -> str:
        """Generate a field encryption key for a specific field"""
        try:
            # Get the table's data key
            table_key = self.get_data_key(table_key_id)

            # Generate a unique key ID
            key_id = f"field_{table_key_id}_{field_name}"

            # Derive the field key from table key using HKDF
            field_key = self._derive_key(table_key, f"field_key_{field_name}", 32)

            # Encrypt the field key with table key
            encrypted_field_key = self._encrypt_data(field_key, table_key)

            # Calculate expiration (30 days from now)
            expires_at = datetime.utcnow() + timedelta(days=30)
            next_rotation_at = datetime.utcnow() + timedelta(days=30)

            # Store the encrypted field key
            with db_manager.get_db() as session:
                field_key_record = EncryptionKey(
                    key_id=key_id,
                    key_type='field',
                    name=f"Field Key for {field_name}",
                    description=f"Field encryption key for field: {field_name} in table: {table_key_id}",
                    encrypted_key=encrypted_field_key,
                    key_hash=self._calculate_key_hash(field_key),
                    algorithm='aes-256-gcm',
                    parent_key_id=self._get_key_record_id(table_key_id),
                    derived_from=f"data_key:{table_key_id}",
                    status='active',
                    security_level='standard',
                    expires_at=expires_at,
                    next_rotation_at=next_rotation_at,
                    created_by='system'
                )
                session.add(field_key_record)
                session.commit()

            # Audit the key generation
            self._audit_key_access(key_id, 'store', 'system', f'Field key generated for field: {field_name}')

            logger.info(f"Generated field key: {key_id} for field: {field_name}")
            return key_id

        except Exception as e:
            logger.error(f"Error generating field key for {field_name}: {e}")
            raise

    def get_data_key(self, key_id: str) -> bytes:
        """Retrieve and decrypt a data encryption key"""
        try:
            master_key = self.get_master_key()

            with db_manager.get_db() as session:
                key_record = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id,
                    EncryptionKey.key_type == 'data',
                    EncryptionKey.status == 'active'
                ).first()

                if not key_record:
                    raise ValueError(f"Data key not found: {key_id}")

                # Check expiration
                if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                    raise ValueError(f"Data key expired: {key_id}")

                # Decrypt the data key
                data_key = self._decrypt_data(key_record.encrypted_key, master_key)

                # Verify integrity
                if self._calculate_key_hash(data_key) != key_record.key_hash:
                    raise ValueError("Data key integrity check failed")

                # Audit the access
                self._audit_key_access(key_id, 'retrieve', 'system', 'Data key retrieved')

                return data_key

        except Exception as e:
            logger.error(f"Error retrieving data key {key_id}: {e}")
            raise

    def get_field_key(self, key_id: str) -> bytes:
        """Retrieve and decrypt a field encryption key"""
        try:
            # Get the parent data key
            with db_manager.get_db() as session:
                key_record = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id,
                    EncryptionKey.key_type == 'field',
                    EncryptionKey.status == 'active'
                ).first()

                if not key_record:
                    raise ValueError(f"Field key not found: {key_id}")

                # Check expiration
                if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                    raise ValueError(f"Field key expired: {key_id}")

                # Get parent data key
                parent_key_id = key_record.derived_from.replace('data_key:', '')
                data_key = self.get_data_key(parent_key_id)

                # Decrypt the field key
                field_key = self._decrypt_data(key_record.encrypted_key, data_key)

                # Verify integrity
                if self._calculate_key_hash(field_key) != key_record.key_hash:
                    raise ValueError("Field key integrity check failed")

                # Audit the access
                self._audit_key_access(key_id, 'retrieve', 'system', 'Field key retrieved')

                return field_key

        except Exception as e:
            logger.error(f"Error retrieving field key {key_id}: {e}")
            raise

    def rotate_key(self, key_id: str, reason: str = "scheduled") -> KeyRotationResult:
        """Rotate an encryption key"""
        start_time = datetime.utcnow()

        try:
            with db_manager.get_db() as session:
                # Get the current key
                key_record = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id,
                    EncryptionKey.status == 'active'
                ).first()

                if not key_record:
                    raise ValueError(f"Key not found for rotation: {key_id}")

                old_key_hash = key_record.key_hash

                # Generate new key based on type
                if key_record.key_type == 'master':
                    new_key = secrets.token_bytes(32)
                    encrypted_new_key = self._encrypt_with_passphrase(new_key, self.master_key_passphrase)
                elif key_record.key_type == 'data':
                    master_key = self.get_master_key()
                    table_name = key_record.key_id.replace('data_', '').split('_')[0]
                    new_key = self._derive_key(master_key, f"data_key_{table_name}", 32)
                    encrypted_new_key = self._encrypt_data(new_key, master_key)
                elif key_record.key_type == 'field':
                    parent_key_id = key_record.derived_from.replace('data_key:', '')
                    data_key = self.get_data_key(parent_key_id)
                    field_name = key_record.key_id.split('_')[-1]
                    new_key = self._derive_key(data_key, f"field_key_{field_name}", 32)
                    encrypted_new_key = self._encrypt_data(new_key, data_key)
                else:
                    raise ValueError(f"Unsupported key type for rotation: {key_record.key_type}")

                # Generate new key ID
                new_key_id = f"{key_record.key_id}_rotated_{secrets.token_hex(4)}"

                # Create new key record
                new_key_record = EncryptionKey(
                    key_id=new_key_id,
                    key_type=key_record.key_type,
                    name=f"{key_record.name} (Rotated)",
                    description=f"Rotated version of {key_record.key_id}",
                    encrypted_key=encrypted_new_key,
                    key_hash=self._calculate_key_hash(new_key),
                    algorithm=key_record.algorithm,
                    parent_key_id=key_record.parent_key_id,
                    derived_from=key_record.derived_from,
                    status='active',
                    security_level=key_record.security_level,
                    expires_at=key_record.expires_at,
                    next_rotation_at=key_record.next_rotation_at,
                    created_by='system'
                )

                # Update old key
                key_record.status = 'deprecated'
                key_record.last_rotated_at = datetime.utcnow()
                key_record.rotation_count += 1
                key_record.updated_at = datetime.utcnow()

                session.add(new_key_record)
                session.commit()

                # Record rotation history
                rotation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                rotation_history = KeyRotationHistory(
                    key_id=key_record.id,
                    old_key_hash=old_key_hash,
                    new_key_hash=new_key_record.key_hash,
                    rotation_reason=reason,
                    rotated_by='system',
                    rotation_time_ms=int(rotation_time),
                    success=True,
                    old_expires_at=key_record.expires_at,
                    new_expires_at=new_key_record.expires_at
                )
                session.add(rotation_history)
                session.commit()

                # Audit the rotation
                self._audit_key_access(key_id, 'rotate', 'system', f'Key rotated: {reason}')

                # Emit event
                self.event_bus.emit(
                    EventChannel.SYSTEM_ALERT,
                    {
                        "type": "key_rotated",
                        "old_key_id": key_id,
                        "new_key_id": new_key_id,
                        "key_type": key_record.key_type,
                        "reason": reason
                    }
                )

                logger.info(f"Successfully rotated key {key_id} to {new_key_id}")

                return KeyRotationResult(
                    success=True,
                    old_key_id=key_id,
                    new_key_id=new_key_id,
                    rotation_time_ms=int(rotation_time)
                )

        except Exception as e:
            logger.error(f"Error rotating key {key_id}: {e}")

            # Record failed rotation
            try:
                with db_manager.get_db() as session:
                    rotation_history = KeyRotationHistory(
                        key_id=self._get_key_record_id(key_id),
                        rotation_reason=reason,
                        rotated_by='system',
                        rotation_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                        success=False,
                        error_message=str(e)
                    )
                    session.add(rotation_history)
                    session.commit()
            except:
                pass  # Don't let audit failure break the main error

            return KeyRotationResult(
                success=False,
                old_key_id=key_id,
                new_key_id="",
                rotation_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                error_message=str(e)
            )

    def get_keys_due_for_rotation(self) -> List[Dict[str, Any]]:
        """Get all keys that are due for rotation"""
        try:
            with db_manager.get_db() as session:
                keys = session.query(EncryptionKey).filter(
                    EncryptionKey.status == 'active',
                    EncryptionKey.next_rotation_at.isnot(None),
                    EncryptionKey.next_rotation_at <= datetime.utcnow()
                ).all()

                result = []
                for key in keys:
                    days_until = 0
                    if key.next_rotation_at:
                        days_until = (key.next_rotation_at - datetime.utcnow()).days

                    result.append({
                        'id': str(key.id),
                        'key_id': key.key_id,
                        'key_type': key.key_type,
                        'name': key.name,
                        'next_rotation_at': key.next_rotation_at.isoformat() if key.next_rotation_at else None,
                        'days_until_rotation': max(0, days_until),
                        'rotation_count': key.rotation_count
                    })

                return result

        except Exception as e:
            logger.error(f"Error getting keys due for rotation: {e}")
            return []

    def _encrypt_with_passphrase(self, data: bytes, passphrase: str) -> bytes:
        """Encrypt data using a passphrase (for master key storage)"""
        # Derive key from passphrase
        salt = secrets.token_bytes(16)
        key = self._derive_key_from_passphrase(passphrase, salt, 32)

        # Generate nonce
        nonce = secrets.token_bytes(12)

        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return salt + nonce + tag + ciphertext
        return salt + nonce + encryptor.tag + ciphertext

    def _decrypt_with_passphrase(self, encrypted_data: bytes, passphrase: str) -> bytes:
        """Decrypt data encrypted with a passphrase"""
        # Extract components
        salt = encrypted_data[:16]
        nonce = encrypted_data[16:28]
        tag = encrypted_data[28:44]
        ciphertext = encrypted_data[44:]

        # Derive key from passphrase
        key = self._derive_key_from_passphrase(passphrase, salt, 32)

        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-GCM"""
        nonce = secrets.token_bytes(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return nonce + encryptor.tag + ciphertext

    def _decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM"""
        nonce = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _derive_key(self, master_key: bytes, info: str, length: int = 32) -> bytes:
        """Derive a key using HKDF"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,
            info=info.encode(),
            backend=self.backend
        )
        return hkdf.derive(master_key)

    def _derive_key_from_passphrase(self, passphrase: str, salt: bytes, length: int = 32) -> bytes:
        """Derive a key from a passphrase using PBKDF2"""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(passphrase.encode())

    def _calculate_key_hash(self, key: bytes) -> str:
        """Calculate SHA256 hash of a key for integrity verification"""
        return hashlib.sha256(key).hexdigest()

    def _get_master_key_record_id(self) -> Optional[str]:
        """Get the database ID of the master key record"""
        try:
            with db_manager.get_db() as session:
                master_key = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == self.master_key_id,
                    EncryptionKey.key_type == 'master'
                ).first()
                return str(master_key.id) if master_key else None
        except:
            return None

    def _get_key_record_id(self, key_id: str) -> Optional[str]:
        """Get the database ID of a key record"""
        try:
            with db_manager.get_db() as session:
                key = session.query(EncryptionKey).filter(
                    EncryptionKey.key_id == key_id
                ).first()
                return str(key.id) if key else None
        except:
            return None

    def _audit_key_access(self, key_id: str, access_type: str, accessor_type: str,
                         details: str = "", success: bool = True, error_message: str = ""):
        """Audit key access operations"""
        try:
            with db_manager.get_db() as session:
                audit_record = KeyAccessAudit(
                    key_id=self._get_key_record_id(key_id),
                    access_type=access_type,
                    accessor_type=accessor_type,
                    success=success,
                    error_message=error_message if not success else None,
                    metadata={"details": details}
                )
                session.add(audit_record)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to audit key access: {e}")

    def __del__(self):
        """Cleanup event bus on destruction"""
        try:
            if hasattr(self, 'event_bus'):
                self.event_bus.stop()
        except:
            pass


# Global instance
key_management_service = KeyManagementService()
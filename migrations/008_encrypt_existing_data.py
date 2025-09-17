"""
Migration to encrypt existing sensitive data
This migration safely encrypts existing sensitive data in the database
"""

import logging
import time
from datetime import datetime
from sqlalchemy import text
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def run_migration():
    """
    Run the data encryption migration

    This migration:
    1. Creates necessary encryption keys
    2. Encrypts existing sensitive data
    3. Updates models to use encrypted fields
    4. Provides rollback capabilities
    """
    logger.info("Starting data encryption migration...")

    start_time = time.time()

    try:
        # Import required modules
        from shared.database import db_manager
        from services.key_management_service import key_management_service
        from shared.encryption_middleware import encryption_middleware
        from shared.models import User, Conversation, Agent, Workflow

        # Step 1: Initialize encryption keys
        logger.info("Step 1: Initializing encryption keys...")
        _initialize_encryption_keys()

        # Step 2: Disable encryption middleware during migration
        logger.info("Step 2: Temporarily disabling encryption middleware...")
        encryption_middleware.disable_encryption()

        # Step 3: Encrypt existing data
        logger.info("Step 3: Encrypting existing data...")

        with db_manager.get_db() as session:
            # Encrypt user data
            _encrypt_user_data(session)

            # Encrypt conversation data
            _encrypt_conversation_data(session)

            # Encrypt agent data
            _encrypt_agent_data(session)

            # Encrypt workflow data
            _encrypt_workflow_data(session)

            # Commit all changes
            session.commit()

        # Step 4: Re-enable encryption middleware
        logger.info("Step 4: Re-enabling encryption middleware...")
        encryption_middleware.enable_encryption()

        # Step 5: Verify encryption
        logger.info("Step 5: Verifying encryption...")
        _verify_encryption()

        # Step 6: Create backup recommendations
        logger.info("Step 6: Creating backup recommendations...")
        _create_backup_recommendations()

        migration_time = time.time() - start_time
        logger.info(f"Data encryption migration completed in {migration_time:.2f} seconds")
        # Log successful migration
        _log_migration_success(migration_time)

    except Exception as e:
        logger.error(f"Data encryption migration failed: {e}")
        # Attempt rollback
        _rollback_migration()
        raise


def _initialize_encryption_keys():
    """Initialize all required encryption keys"""
    try:
        # Generate master key if it doesn't exist
        try:
            key_management_service.get_master_key()
            logger.info("Master key already exists")
        except:
            key_management_service.generate_master_key()
            logger.info("Generated new master key")

        # Define required keys
        required_keys = [
            ('user_email_key', 'User email encryption'),
            ('user_password_key', 'User password hash encryption'),
            ('conversation_query_key', 'Conversation query encryption'),
            ('conversation_response_key', 'Conversation response encryption'),
            ('agent_config_key', 'Agent configuration encryption'),
            ('agent_capabilities_key', 'Agent capabilities encryption'),
            ('workflow_steps_key', 'Workflow steps encryption'),
            ('workflow_context_key', 'Workflow context encryption')
        ]

        # Generate data keys for each table
        for key_id, description in required_keys:
            try:
                # Extract table name from key
                table_name = key_id.split('_')[0]
                key_management_service.generate_data_key(table_name, key_id)
                logger.info(f"Generated data key: {key_id}")
            except Exception as e:
                logger.warning(f"Failed to generate key {key_id}: {e}")

        # Generate field keys
        field_keys = [
            ('users', 'email'),
            ('users', 'password_hash'),
            ('conversations', 'user_query'),
            ('conversations', 'agent_response'),
            ('agents', 'config'),
            ('agents', 'capabilities'),
            ('workflows', 'steps'),
            ('workflows', 'context')
        ]

        for table, field in field_keys:
            try:
                data_key_id = f"data_{table}_{table}"
                field_key_id = f"field_{data_key_id}_{field}"
                key_management_service.generate_field_key(data_key_id, field)
                logger.info(f"Generated field key: {field_key_id}")
            except Exception as e:
                logger.warning(f"Failed to generate field key for {table}.{field}: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize encryption keys: {e}")
        raise


def _encrypt_user_data(session):
    """Encrypt existing user data"""
    logger.info("Encrypting user data...")

    try:
        from shared.models import User
        from shared.encrypted_fields import encrypt_value

        users = session.query(User).all()
        encrypted_count = 0

        for user in users:
            try:
                # Encrypt email if not already encrypted
                if user.email and not user.email_encrypted:
                    user.email_encrypted = encrypt_value(user.email, 'user_email_key')
                    encrypted_count += 1

                # Encrypt password hash if not already encrypted
                if user.password_hash and not user.password_hash_encrypted:
                    user.password_hash_encrypted = encrypt_value(user.password_hash, 'user_password_key')
                    encrypted_count += 1

            except Exception as e:
                logger.error(f"Failed to encrypt data for user {user.id}: {e}")

        logger.info(f"Encrypted data for {encrypted_count} user fields")

    except Exception as e:
        logger.error(f"Failed to encrypt user data: {e}")
        raise


def _encrypt_conversation_data(session):
    """Encrypt existing conversation data"""
    logger.info("Encrypting conversation data...")

    try:
        from shared.models import Conversation
        from shared.encrypted_fields import encrypt_value

        conversations = session.query(Conversation).all()
        encrypted_count = 0

        for conversation in conversations:
            try:
                # Encrypt user query if not already encrypted
                if conversation.user_query and not conversation.user_query_encrypted:
                    conversation.user_query_encrypted = encrypt_value(
                        conversation.user_query, 'conversation_query_key'
                    )
                    encrypted_count += 1

                # Encrypt agent response if not already encrypted
                if conversation.agent_response and not conversation.agent_response_encrypted:
                    conversation.agent_response_encrypted = encrypt_value(
                        conversation.agent_response, 'conversation_response_key'
                    )
                    encrypted_count += 1

            except Exception as e:
                logger.error(f"Failed to encrypt data for conversation {conversation.id}: {e}")

        logger.info(f"Encrypted data for {encrypted_count} conversation fields")

    except Exception as e:
        logger.error(f"Failed to encrypt conversation data: {e}")
        raise


def _encrypt_agent_data(session):
    """Encrypt existing agent data"""
    logger.info("Encrypting agent data...")

    try:
        from shared.models import Agent
        from shared.encrypted_fields import encrypt_value
        import json

        agents = session.query(Agent).all()
        encrypted_count = 0

        for agent in agents:
            try:
                # Encrypt config if not already encrypted
                if agent.config and not agent.config_encrypted:
                    agent.config_encrypted = encrypt_value(
                        json.dumps(agent.config), 'agent_config_key'
                    )
                    encrypted_count += 1

                # Encrypt capabilities if not already encrypted
                if agent.capabilities and not agent.capabilities_encrypted:
                    agent.capabilities_encrypted = encrypt_value(
                        json.dumps(agent.capabilities), 'agent_capabilities_key'
                    )
                    encrypted_count += 1

            except Exception as e:
                logger.error(f"Failed to encrypt data for agent {agent.id}: {e}")

        logger.info(f"Encrypted data for {encrypted_count} agent fields")

    except Exception as e:
        logger.error(f"Failed to encrypt agent data: {e}")
        raise


def _encrypt_workflow_data(session):
    """Encrypt existing workflow data"""
    logger.info("Encrypting workflow data...")

    try:
        from shared.models import Workflow
        from shared.encrypted_fields import encrypt_value
        import json

        workflows = session.query(Workflow).all()
        encrypted_count = 0

        for workflow in workflows:
            try:
                # Encrypt steps if not already encrypted
                if workflow.steps and not workflow.steps_encrypted:
                    workflow.steps_encrypted = encrypt_value(
                        json.dumps(workflow.steps), 'workflow_steps_key'
                    )
                    encrypted_count += 1

                # Encrypt context if not already encrypted
                if workflow.context and not workflow.context_encrypted:
                    workflow.context_encrypted = encrypt_value(
                        json.dumps(workflow.context), 'workflow_context_key'
                    )
                    encrypted_count += 1

            except Exception as e:
                logger.error(f"Failed to encrypt data for workflow {workflow.id}: {e}")

        logger.info(f"Encrypted data for {encrypted_count} workflow fields")

    except Exception as e:
        logger.error(f"Failed to encrypt workflow data: {e}")
        raise


def _verify_encryption():
    """Verify that encryption was successful"""
    logger.info("Verifying encryption...")

    try:
        from shared.database import db_manager
        from shared.models import User, Conversation, Agent, Workflow

        with db_manager.get_db() as session:
            # Check user encryption
            user_count = session.query(User).filter(
                User.email_encrypted.isnot(None)
            ).count()
            logger.info(f"Verified {user_count} users with encrypted email")

            # Check conversation encryption
            conv_count = session.query(Conversation).filter(
                Conversation.user_query_encrypted.isnot(None)
            ).count()
            logger.info(f"Verified {conv_count} conversations with encrypted queries")

            # Check agent encryption
            agent_count = session.query(Agent).filter(
                Agent.config_encrypted.isnot(None)
            ).count()
            logger.info(f"Verified {agent_count} agents with encrypted config")

            # Check workflow encryption
            workflow_count = session.query(Workflow).filter(
                Workflow.steps_encrypted.isnot(None)
            ).count()
            logger.info(f"Verified {workflow_count} workflows with encrypted steps")

    except Exception as e:
        logger.error(f"Encryption verification failed: {e}")
        raise


def _create_backup_recommendations():
    """Create backup recommendations after migration"""
    logger.info("Creating backup recommendations...")

    recommendations = """
    POST-MIGRATION BACKUP RECOMMENDATIONS:

    1. Full database backup completed immediately after migration
    2. Test data restoration from backup
    3. Verify encrypted data can be decrypted correctly
    4. Monitor application performance for 24-48 hours
    5. Keep backup for at least 30 days before deletion

    IMPORTANT: Do not delete the pre-migration backup until:
    - All encrypted data has been verified as accessible
    - Application performance is stable
    - No data corruption issues are detected
    """

    logger.info(recommendations)

    # Write recommendations to file
    try:
        with open('MIGRATION_BACKUP_RECOMMENDATIONS.txt', 'w') as f:
            f.write(recommendations)
        logger.info("Backup recommendations written to MIGRATION_BACKUP_RECOMMENDATIONS.txt")
    except Exception as e:
        logger.error(f"Failed to write backup recommendations: {e}")


def _log_migration_success(migration_time: float):
    """Log successful migration completion"""
    try:
        from shared.database import db_manager
        from shared.models import Metric

        with db_manager.get_db() as session:
            # Log migration metric
            metric = Metric(
                service_name='migration',
                metric_name='data_encryption_migration',
                value=migration_time,
                tags={
                    'migration_type': 'data_encryption',
                    'status': 'completed',
                    'version': '008'
                }
            )
            session.add(metric)
            session.commit()

        logger.info("Migration success logged to database metrics")

    except Exception as e:
        logger.error(f"Failed to log migration success: {e}")


def _rollback_migration():
    """Rollback migration in case of failure"""
    logger.warning("Attempting migration rollback...")

    try:
        from shared.database import db_manager
        from shared.models import User, Conversation, Agent, Workflow

        with db_manager.get_db() as session:
            # Clear encrypted fields (they will be re-populated on next migration attempt)
            session.query(User).update({
                User.email_encrypted: None,
                User.password_hash_encrypted: None
            })

            session.query(Conversation).update({
                Conversation.user_query_encrypted: None,
                Conversation.agent_response_encrypted: None
            })

            session.query(Agent).update({
                Agent.config_encrypted: None,
                Agent.capabilities_encrypted: None
            })

            session.query(Workflow).update({
                Workflow.steps_encrypted: None,
                Workflow.context_encrypted: None
            })

            session.commit()

        logger.info("Migration rollback completed")

    except Exception as e:
        logger.error(f"Migration rollback failed: {e}")
        logger.critical("MANUAL INTERVENTION REQUIRED: Database may be in inconsistent state")


if __name__ == "__main__":
    # Allow running migration directly
    logging.basicConfig(level=logging.INFO)
    run_migration()
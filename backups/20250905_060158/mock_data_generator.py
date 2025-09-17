#!/usr/bin/env python3
"""
Mock Data Generator for Testing
Generates realistic mock data for various data types and scenarios
"""

import os
import sys
import json
import random
import string
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta, date
import hashlib
import base64
from faker import Faker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MockDataType(Enum):
    """Types of mock data"""
    # Personal Data
    PERSON = "person"
    ADDRESS = "address"
    CONTACT = "contact"
    COMPANY = "company"
    
    # Authentication
    USER = "user"
    CREDENTIAL = "credential"
    TOKEN = "token"
    SESSION = "session"
    
    # Financial
    PAYMENT = "payment"
    TRANSACTION = "transaction"
    INVOICE = "invoice"
    ACCOUNT = "account"
    
    # Product/E-commerce
    PRODUCT = "product"
    ORDER = "order"
    CART = "cart"
    REVIEW = "review"
    
    # Content
    ARTICLE = "article"
    COMMENT = "comment"
    POST = "post"
    MESSAGE = "message"
    
    # Technical
    API_RESPONSE = "api_response"
    ERROR = "error"
    LOG_ENTRY = "log_entry"
    METRIC = "metric"
    
    # Files/Media
    FILE = "file"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"


class DataPattern(Enum):
    """Common data patterns"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    CYCLIC = "cyclic"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass
class MockConfig:
    """Configuration for mock data generation"""
    locale: str = "en_US"
    seed: Optional[int] = None
    deterministic: bool = False
    realistic: bool = True
    include_nulls: bool = False
    null_probability: float = 0.1
    error_probability: float = 0.0
    include_edge_cases: bool = True
    date_range_days: int = 365
    min_array_size: int = 0
    max_array_size: int = 10


@dataclass
class MockSchema:
    """Schema definition for complex mock data"""
    fields: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    sample_size: int = 1


class MockDataGenerator:
    """Main mock data generator"""
    
    def __init__(self, config: MockConfig = None):
        self.config = config or MockConfig()
        self.faker = Faker(self.config.locale)
        
        if self.config.seed:
            random.seed(self.config.seed)
            self.faker.seed_instance(self.config.seed)
        
        self.generators = self._init_generators()
        self.cache = {}
        
    def _init_generators(self) -> Dict[str, callable]:
        """Initialize data type generators"""
        return {
            MockDataType.PERSON: self._generate_person,
            MockDataType.ADDRESS: self._generate_address,
            MockDataType.CONTACT: self._generate_contact,
            MockDataType.COMPANY: self._generate_company,
            MockDataType.USER: self._generate_user,
            MockDataType.CREDENTIAL: self._generate_credential,
            MockDataType.TOKEN: self._generate_token,
            MockDataType.SESSION: self._generate_session,
            MockDataType.PAYMENT: self._generate_payment,
            MockDataType.TRANSACTION: self._generate_transaction,
            MockDataType.INVOICE: self._generate_invoice,
            MockDataType.ACCOUNT: self._generate_account,
            MockDataType.PRODUCT: self._generate_product,
            MockDataType.ORDER: self._generate_order,
            MockDataType.CART: self._generate_cart,
            MockDataType.REVIEW: self._generate_review,
            MockDataType.ARTICLE: self._generate_article,
            MockDataType.COMMENT: self._generate_comment,
            MockDataType.POST: self._generate_post,
            MockDataType.MESSAGE: self._generate_message,
            MockDataType.API_RESPONSE: self._generate_api_response,
            MockDataType.ERROR: self._generate_error,
            MockDataType.LOG_ENTRY: self._generate_log_entry,
            MockDataType.METRIC: self._generate_metric,
            MockDataType.FILE: self._generate_file,
            MockDataType.IMAGE: self._generate_image,
            MockDataType.VIDEO: self._generate_video,
            MockDataType.DOCUMENT: self._generate_document
        }
    
    def generate(
        self,
        data_type: MockDataType,
        count: int = 1,
        **kwargs
    ) -> Union[Dict, List[Dict]]:
        """Generate mock data of specified type"""
        if data_type in self.generators:
            generator = self.generators[data_type]
            
            if count == 1:
                return generator(**kwargs)
            else:
                return [generator(**kwargs) for _ in range(count)]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def generate_from_schema(self, schema: MockSchema) -> Union[Dict, List[Dict]]:
        """Generate data based on schema definition"""
        results = []
        
        for _ in range(schema.sample_size):
            data = {}
            
            # Generate each field
            for field_name, field_type in schema.fields.items():
                # Check if should be null
                if self._should_be_null(field_name, schema.required_fields):
                    data[field_name] = None
                else:
                    data[field_name] = self._generate_field_value(
                        field_type,
                        field_name,
                        schema.constraints.get(field_name, {})
                    )
            
            # Apply relationships
            for field, related_to in schema.relationships.items():
                if related_to in data and field in data:
                    # Make fields related (e.g., same value or derived)
                    data[field] = self._derive_value(data[field], data[related_to])
            
            results.append(data)
        
        return results if schema.sample_size > 1 else results[0]
    
    def _should_be_null(self, field_name: str, required_fields: List[str]) -> bool:
        """Determine if field should be null"""
        if field_name in required_fields:
            return False
        
        if self.config.include_nulls:
            return random.random() < self.config.null_probability
        
        return False
    
    def _generate_field_value(
        self,
        field_type: str,
        field_name: str,
        constraints: Dict
    ) -> Any:
        """Generate value for a field based on type"""
        # Handle basic types
        if field_type == "string":
            return self._generate_string_field(field_name, constraints)
        elif field_type == "integer":
            return self._generate_integer_field(constraints)
        elif field_type == "float":
            return self._generate_float_field(constraints)
        elif field_type == "boolean":
            return random.choice([True, False])
        elif field_type == "date":
            return self._generate_date_field(constraints)
        elif field_type == "datetime":
            return self._generate_datetime_field(constraints)
        elif field_type == "uuid":
            return str(uuid.uuid4())
        elif field_type == "email":
            return self.faker.email()
        elif field_type == "url":
            return self.faker.url()
        elif field_type == "array":
            return self._generate_array_field(constraints)
        elif field_type == "object":
            return self._generate_object_field(constraints)
        else:
            # Try to use mock data type
            try:
                mock_type = MockDataType(field_type)
                return self.generate(mock_type)
            except:
                return f"mock_{field_type}_{random.randint(1, 1000)}"
    
    def _generate_string_field(self, field_name: str, constraints: Dict) -> str:
        """Generate string based on field name and constraints"""
        min_length = constraints.get("min_length", 1)
        max_length = constraints.get("max_length", 100)
        pattern = constraints.get("pattern")
        
        # Smart generation based on field name
        field_lower = field_name.lower()
        
        if "name" in field_lower:
            if "first" in field_lower:
                return self.faker.first_name()
            elif "last" in field_lower:
                return self.faker.last_name()
            elif "company" in field_lower:
                return self.faker.company()
            else:
                return self.faker.name()
        elif "email" in field_lower:
            return self.faker.email()
        elif "phone" in field_lower:
            return self.faker.phone_number()
        elif "address" in field_lower:
            return self.faker.address()
        elif "city" in field_lower:
            return self.faker.city()
        elif "country" in field_lower:
            return self.faker.country()
        elif "description" in field_lower or "text" in field_lower:
            return self.faker.text(max_nb_chars=max_length)
        elif "title" in field_lower:
            return self.faker.sentence(nb_words=6)
        elif "url" in field_lower or "link" in field_lower:
            return self.faker.url()
        elif "password" in field_lower:
            return self._generate_password()
        elif "token" in field_lower:
            return self._generate_token()
        elif "id" in field_lower:
            return self._generate_id()
        else:
            # Generic string
            if pattern:
                return self._generate_from_pattern(pattern)
            else:
                length = random.randint(max(min_length, 5), max_length)  # faker.text requires min 5 chars
                return self.faker.text(max_nb_chars=length)
    
    def _generate_integer_field(self, constraints: Dict) -> int:
        """Generate integer with constraints"""
        min_val = constraints.get("min", 0)
        max_val = constraints.get("max", 1000000)
        
        if self.config.include_edge_cases and random.random() < 0.1:
            # Return edge case
            return random.choice([min_val, max_val, 0, -1, 1])
        
        return random.randint(min_val, max_val)
    
    def _generate_float_field(self, constraints: Dict) -> float:
        """Generate float with constraints"""
        min_val = constraints.get("min", 0.0)
        max_val = constraints.get("max", 1000000.0)
        precision = constraints.get("precision", 2)
        
        if self.config.include_edge_cases and random.random() < 0.1:
            # Return edge case
            return random.choice([min_val, max_val, 0.0, -0.0])
        
        value = random.uniform(min_val, max_val)
        return round(value, precision)
    
    def _generate_date_field(self, constraints: Dict) -> str:
        """Generate date with constraints"""
        start_date = constraints.get("start", date.today() - timedelta(days=365))
        end_date = constraints.get("end", date.today() + timedelta(days=365))
        
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date).date()
        
        fake_date = self.faker.date_between(start_date, end_date)
        return fake_date.isoformat()
    
    def _generate_datetime_field(self, constraints: Dict) -> str:
        """Generate datetime with constraints"""
        start_date = constraints.get("start", datetime.now() - timedelta(days=365))
        end_date = constraints.get("end", datetime.now() + timedelta(days=365))
        
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        fake_datetime = self.faker.date_time_between(start_date, end_date)
        return fake_datetime.isoformat()
    
    def _generate_array_field(self, constraints: Dict) -> List[Any]:
        """Generate array with constraints"""
        min_size = constraints.get("min_size", self.config.min_array_size)
        max_size = constraints.get("max_size", self.config.max_array_size)
        item_type = constraints.get("item_type", "string")
        
        size = random.randint(min_size, max_size)
        array = []
        
        for _ in range(size):
            item_constraints = constraints.get("item_constraints", {})
            array.append(self._generate_field_value(item_type, "item", item_constraints))
        
        return array
    
    def _generate_object_field(self, constraints: Dict) -> Dict[str, Any]:
        """Generate nested object"""
        properties = constraints.get("properties", {})
        obj = {}
        
        for prop_name, prop_type in properties.items():
            obj[prop_name] = self._generate_field_value(prop_type, prop_name, {})
        
        return obj
    
    def _generate_person(self, **kwargs) -> Dict[str, Any]:
        """Generate person data"""
        person = {
            "id": str(uuid.uuid4()),
            "first_name": self.faker.first_name(),
            "last_name": self.faker.last_name(),
            "email": self.faker.email(),
            "phone": self.faker.phone_number(),
            "date_of_birth": self.faker.date_of_birth().isoformat(),
            "gender": random.choice(["male", "female", "other"]),
            "nationality": self.faker.country(),
            "occupation": self.faker.job(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        if self.config.realistic:
            # Add consistent email based on name
            person["email"] = f"{person['first_name'].lower()}.{person['last_name'].lower()}@{self.faker.domain_name()}"
        
        return person
    
    def _generate_address(self, **kwargs) -> Dict[str, Any]:
        """Generate address data"""
        return {
            "id": str(uuid.uuid4()),
            "street": self.faker.street_address(),
            "city": self.faker.city(),
            "state": self.faker.state(),
            "country": self.faker.country(),
            "postal_code": self.faker.postcode(),
            "latitude": float(self.faker.latitude()),
            "longitude": float(self.faker.longitude()),
            "type": random.choice(["home", "work", "billing", "shipping"])
        }
    
    def _generate_contact(self, **kwargs) -> Dict[str, Any]:
        """Generate contact information"""
        return {
            "id": str(uuid.uuid4()),
            "name": self.faker.name(),
            "email": self.faker.email(),
            "phone": self.faker.phone_number(),
            "mobile": self.faker.phone_number(),
            "fax": self.faker.phone_number() if random.random() > 0.5 else None,
            "website": self.faker.url(),
            "social_media": {
                "twitter": f"@{self.faker.user_name()}",
                "linkedin": f"linkedin.com/in/{self.faker.user_name()}",
                "github": f"github.com/{self.faker.user_name()}"
            }
        }
    
    def _generate_company(self, **kwargs) -> Dict[str, Any]:
        """Generate company data"""
        return {
            "id": str(uuid.uuid4()),
            "name": self.faker.company(),
            "industry": self.faker.bs(),
            "description": self.faker.catch_phrase(),
            "founded": self.faker.date_between("-30y", "-1y").isoformat(),
            "employees": random.randint(1, 100000),
            "revenue": round(random.uniform(10000, 1000000000), 2),
            "website": self.faker.url(),
            "email": f"info@{self.faker.domain_name()}",
            "phone": self.faker.phone_number(),
            "address": self._generate_address(),
            "ceo": self.faker.name(),
            "stock_symbol": ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 4)))
        }
    
    def _generate_user(self, **kwargs) -> Dict[str, Any]:
        """Generate user account data"""
        username = self.faker.user_name()
        return {
            "id": str(uuid.uuid4()),
            "username": username,
            "email": self.faker.email(),
            "password_hash": hashlib.sha256(self.faker.password().encode()).hexdigest(),
            "first_name": self.faker.first_name(),
            "last_name": self.faker.last_name(),
            "avatar": self.faker.image_url(),
            "bio": self.faker.text(max_nb_chars=200),
            "role": random.choice(["admin", "user", "moderator", "guest"]),
            "status": random.choice(["active", "inactive", "suspended", "pending"]),
            "email_verified": random.choice([True, False]),
            "two_factor_enabled": random.choice([True, False]),
            "last_login": self.faker.date_time_between("-30d", "now").isoformat(),
            "created_at": self.faker.date_time_between("-2y", "-1d").isoformat(),
            "updated_at": datetime.now().isoformat(),
            "preferences": {
                "theme": random.choice(["light", "dark", "auto"]),
                "language": random.choice(["en", "es", "fr", "de", "zh"]),
                "notifications": random.choice([True, False])
            }
        }
    
    def _generate_credential(self, **kwargs) -> Dict[str, Any]:
        """Generate authentication credentials"""
        return {
            "username": self.faker.user_name(),
            "password": self._generate_password(),
            "api_key": self._generate_api_key(),
            "secret": self._generate_secret(),
            "access_token": self._generate_token(),
            "refresh_token": self._generate_token(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    
    def _generate_token(self, **kwargs) -> str:
        """Generate authentication token"""
        token_bytes = os.urandom(32)
        return base64.urlsafe_b64encode(token_bytes).decode('utf-8').rstrip('=')
    
    def _generate_session(self, **kwargs) -> Dict[str, Any]:
        """Generate session data"""
        return {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "token": self._generate_token(),
            "ip_address": self.faker.ipv4(),
            "user_agent": self.faker.user_agent(),
            "device": random.choice(["desktop", "mobile", "tablet"]),
            "browser": random.choice(["chrome", "firefox", "safari", "edge"]),
            "os": random.choice(["windows", "macos", "linux", "ios", "android"]),
            "location": {
                "country": self.faker.country_code(),
                "city": self.faker.city(),
                "timezone": self.faker.timezone()
            },
            "created_at": self.faker.date_time_between("-1d", "now").isoformat(),
            "last_activity": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
    
    def _generate_payment(self, **kwargs) -> Dict[str, Any]:
        """Generate payment data"""
        return {
            "id": str(uuid.uuid4()),
            "amount": round(random.uniform(0.99, 9999.99), 2),
            "currency": random.choice(["USD", "EUR", "GBP", "JPY"]),
            "method": random.choice(["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]),
            "status": random.choice(["pending", "completed", "failed", "refunded"]),
            "card_last_four": f"{random.randint(1000, 9999)}",
            "card_brand": random.choice(["visa", "mastercard", "amex", "discover"]),
            "billing_address": self._generate_address(),
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "reference": f"ref_{random.randint(100000, 999999)}",
            "description": self.faker.sentence(),
            "metadata": {
                "order_id": str(uuid.uuid4()),
                "customer_id": str(uuid.uuid4()),
                "invoice_number": f"INV-{random.randint(1000, 9999)}"
            },
            "created_at": datetime.now().isoformat()
        }
    
    def _generate_transaction(self, **kwargs) -> Dict[str, Any]:
        """Generate financial transaction"""
        return {
            "id": str(uuid.uuid4()),
            "type": random.choice(["debit", "credit", "transfer", "withdrawal", "deposit"]),
            "amount": round(random.uniform(-5000, 5000), 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "from_account": f"ACC{random.randint(10000000, 99999999)}",
            "to_account": f"ACC{random.randint(10000000, 99999999)}",
            "status": random.choice(["pending", "completed", "failed", "cancelled"]),
            "description": self.faker.sentence(),
            "reference": f"TXN{random.randint(100000, 999999)}",
            "timestamp": datetime.now().isoformat(),
            "balance_after": round(random.uniform(0, 100000), 2)
        }
    
    def _generate_invoice(self, **kwargs) -> Dict[str, Any]:
        """Generate invoice data"""
        items = []
        num_items = random.randint(1, 5)
        subtotal = 0
        
        for _ in range(num_items):
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(9.99, 999.99), 2)
            total = round(quantity * unit_price, 2)
            subtotal += total
            
            items.append({
                "description": self.faker.sentence(nb_words=4),
                "quantity": quantity,
                "unit_price": unit_price,
                "total": total
            })
        
        tax = round(subtotal * 0.1, 2)
        total = round(subtotal + tax, 2)
        
        return {
            "id": str(uuid.uuid4()),
            "invoice_number": f"INV-{datetime.now().year}-{random.randint(1000, 9999)}",
            "status": random.choice(["draft", "sent", "paid", "overdue", "cancelled"]),
            "issue_date": date.today().isoformat(),
            "due_date": (date.today() + timedelta(days=30)).isoformat(),
            "customer": self._generate_company(),
            "vendor": self._generate_company(),
            "items": items,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "currency": "USD",
            "payment_terms": random.choice(["net30", "net60", "due_on_receipt"]),
            "notes": self.faker.text(max_nb_chars=200)
        }
    
    def _generate_account(self, **kwargs) -> Dict[str, Any]:
        """Generate account data"""
        return {
            "id": str(uuid.uuid4()),
            "account_number": f"{random.randint(10000000, 99999999)}",
            "account_type": random.choice(["checking", "savings", "credit", "investment"]),
            "balance": round(random.uniform(-1000, 100000), 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "status": random.choice(["active", "inactive", "frozen", "closed"]),
            "owner": self._generate_person(),
            "created_date": self.faker.date_between("-5y", "today").isoformat(),
            "last_activity": datetime.now().isoformat(),
            "interest_rate": round(random.uniform(0, 5), 2),
            "credit_limit": round(random.uniform(1000, 50000), 2) if random.random() > 0.5 else None
        }
    
    def _generate_product(self, **kwargs) -> Dict[str, Any]:
        """Generate product data"""
        return {
            "id": str(uuid.uuid4()),
            "sku": f"SKU-{random.randint(100000, 999999)}",
            "name": self.faker.sentence(nb_words=3),
            "description": self.faker.text(max_nb_chars=500),
            "category": random.choice(["electronics", "clothing", "food", "books", "toys", "sports"]),
            "subcategory": self.faker.word(),
            "brand": self.faker.company(),
            "price": round(random.uniform(0.99, 999.99), 2),
            "cost": round(random.uniform(0.50, 500.00), 2),
            "currency": "USD",
            "weight": round(random.uniform(0.1, 50), 2),
            "dimensions": {
                "length": round(random.uniform(1, 100), 1),
                "width": round(random.uniform(1, 100), 1),
                "height": round(random.uniform(1, 100), 1),
                "unit": "cm"
            },
            "stock": random.randint(0, 1000),
            "status": random.choice(["available", "out_of_stock", "discontinued"]),
            "images": [self.faker.image_url() for _ in range(random.randint(1, 5))],
            "tags": [self.faker.word() for _ in range(random.randint(2, 6))],
            "rating": round(random.uniform(1, 5), 1),
            "reviews_count": random.randint(0, 1000),
            "created_at": self.faker.date_time_between("-2y", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_order(self, **kwargs) -> Dict[str, Any]:
        """Generate order data"""
        items = []
        num_items = random.randint(1, 5)
        subtotal = 0
        
        for _ in range(num_items):
            product = self._generate_product()
            quantity = random.randint(1, 5)
            price = product["price"]
            total = round(quantity * price, 2)
            subtotal += total
            
            items.append({
                "product": product,
                "quantity": quantity,
                "price": price,
                "total": total
            })
        
        shipping = round(random.uniform(0, 50), 2)
        tax = round(subtotal * 0.1, 2)
        total = round(subtotal + shipping + tax, 2)
        
        return {
            "id": str(uuid.uuid4()),
            "order_number": f"ORD-{datetime.now().year}-{random.randint(100000, 999999)}",
            "status": random.choice(["pending", "processing", "shipped", "delivered", "cancelled"]),
            "customer": self._generate_person(),
            "items": items,
            "subtotal": subtotal,
            "shipping": shipping,
            "tax": tax,
            "total": total,
            "currency": "USD",
            "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
            "shipping_address": self._generate_address(),
            "billing_address": self._generate_address(),
            "tracking_number": f"TRACK{random.randint(100000000, 999999999)}",
            "notes": self.faker.text(max_nb_chars=200) if random.random() > 0.5 else None,
            "created_at": self.faker.date_time_between("-30d", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_cart(self, **kwargs) -> Dict[str, Any]:
        """Generate shopping cart data"""
        items = []
        num_items = random.randint(1, 10)
        total = 0
        
        for _ in range(num_items):
            product = self._generate_product()
            quantity = random.randint(1, 5)
            item_total = round(quantity * product["price"], 2)
            total += item_total
            
            items.append({
                "product_id": product["id"],
                "product_name": product["name"],
                "quantity": quantity,
                "price": product["price"],
                "total": item_total,
                "added_at": self.faker.date_time_between("-7d", "now").isoformat()
            })
        
        return {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "items": items,
            "total": round(total, 2),
            "currency": "USD",
            "created_at": self.faker.date_time_between("-7d", "now").isoformat(),
            "updated_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
        }
    
    def _generate_review(self, **kwargs) -> Dict[str, Any]:
        """Generate product review"""
        return {
            "id": str(uuid.uuid4()),
            "product_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "rating": random.randint(1, 5),
            "title": self.faker.sentence(nb_words=6),
            "comment": self.faker.text(max_nb_chars=500),
            "pros": [self.faker.sentence(nb_words=4) for _ in range(random.randint(1, 3))],
            "cons": [self.faker.sentence(nb_words=4) for _ in range(random.randint(0, 2))],
            "verified_purchase": random.choice([True, False]),
            "helpful_count": random.randint(0, 100),
            "images": [self.faker.image_url() for _ in range(random.randint(0, 3))],
            "created_at": self.faker.date_time_between("-1y", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_article(self, **kwargs) -> Dict[str, Any]:
        """Generate article/blog post"""
        return {
            "id": str(uuid.uuid4()),
            "title": self.faker.sentence(nb_words=8),
            "slug": self.faker.slug(),
            "author": self._generate_person(),
            "category": random.choice(["technology", "business", "health", "travel", "food"]),
            "tags": [self.faker.word() for _ in range(random.randint(3, 8))],
            "summary": self.faker.text(max_nb_chars=200),
            "content": self.faker.text(max_nb_chars=5000),
            "featured_image": self.faker.image_url(),
            "status": random.choice(["draft", "published", "archived"]),
            "views": random.randint(0, 10000),
            "likes": random.randint(0, 1000),
            "comments_count": random.randint(0, 100),
            "reading_time": random.randint(1, 20),
            "published_at": self.faker.date_time_between("-1y", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_comment(self, **kwargs) -> Dict[str, Any]:
        """Generate comment data"""
        return {
            "id": str(uuid.uuid4()),
            "post_id": str(uuid.uuid4()),
            "parent_id": str(uuid.uuid4()) if random.random() > 0.7 else None,
            "author": self._generate_person(),
            "content": self.faker.text(max_nb_chars=500),
            "likes": random.randint(0, 100),
            "dislikes": random.randint(0, 50),
            "status": random.choice(["approved", "pending", "spam", "deleted"]),
            "created_at": self.faker.date_time_between("-30d", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_post(self, **kwargs) -> Dict[str, Any]:
        """Generate social media post"""
        return {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "content": self.faker.text(max_nb_chars=280),
            "media": [self.faker.image_url() for _ in range(random.randint(0, 4))],
            "hashtags": [f"#{self.faker.word()}" for _ in range(random.randint(0, 5))],
            "mentions": [f"@{self.faker.user_name()}" for _ in range(random.randint(0, 3))],
            "likes": random.randint(0, 10000),
            "shares": random.randint(0, 1000),
            "comments": random.randint(0, 500),
            "visibility": random.choice(["public", "private", "followers"]),
            "created_at": self.faker.date_time_between("-7d", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_message(self, **kwargs) -> Dict[str, Any]:
        """Generate message/chat data"""
        return {
            "id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "sender_id": str(uuid.uuid4()),
            "recipient_id": str(uuid.uuid4()),
            "content": self.faker.text(max_nb_chars=500),
            "type": random.choice(["text", "image", "video", "file", "voice"]),
            "attachments": [self.faker.file_path() for _ in range(random.randint(0, 3))],
            "status": random.choice(["sent", "delivered", "read", "failed"]),
            "edited": random.choice([True, False]),
            "deleted": False,
            "created_at": self.faker.date_time_between("-7d", "now").isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_api_response(self, **kwargs) -> Dict[str, Any]:
        """Generate API response"""
        success = random.random() > self.config.error_probability
        
        if success:
            return {
                "success": True,
                "status": 200,
                "message": "Request successful",
                "data": {
                    "id": str(uuid.uuid4()),
                    "result": self.faker.sentence(),
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "version": "1.0.0",
                    "request_id": str(uuid.uuid4()),
                    "duration_ms": random.randint(10, 500)
                }
            }
        else:
            return self._generate_error()
    
    def _generate_error(self, **kwargs) -> Dict[str, Any]:
        """Generate error response"""
        error_codes = [400, 401, 403, 404, 409, 422, 500, 502, 503]
        error_messages = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            409: "Conflict",
            422: "Unprocessable Entity",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable"
        }
        
        code = random.choice(error_codes)
        
        return {
            "success": False,
            "error": {
                "code": code,
                "message": error_messages.get(code, "Unknown Error"),
                "details": self.faker.sentence(),
                "timestamp": datetime.now().isoformat(),
                "request_id": str(uuid.uuid4()),
                "path": self.faker.file_path()
            }
        }
    
    def _generate_log_entry(self, **kwargs) -> Dict[str, Any]:
        """Generate log entry"""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "level": random.choice(levels),
            "message": self.faker.sentence(),
            "logger": random.choice(["app", "api", "db", "auth", "payment"]),
            "module": self.faker.file_name(),
            "function": self.faker.word(),
            "line": random.randint(1, 1000),
            "user_id": str(uuid.uuid4()) if random.random() > 0.5 else None,
            "session_id": str(uuid.uuid4()),
            "ip_address": self.faker.ipv4(),
            "extra": {
                "duration_ms": random.randint(1, 5000),
                "memory_mb": random.randint(10, 500),
                "cpu_percent": random.randint(0, 100)
            }
        }
    
    def _generate_metric(self, **kwargs) -> Dict[str, Any]:
        """Generate metric/telemetry data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metric_name": random.choice([
                "cpu_usage", "memory_usage", "disk_usage",
                "request_count", "response_time", "error_rate"
            ]),
            "value": round(random.uniform(0, 100), 2),
            "unit": random.choice(["percent", "bytes", "ms", "count"]),
            "tags": {
                "host": self.faker.hostname(),
                "environment": random.choice(["production", "staging", "development"]),
                "service": random.choice(["api", "web", "worker", "database"]),
                "region": random.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])
            },
            "metadata": {
                "collector": "agent",
                "version": "1.0.0"
            }
        }
    
    def _generate_file(self, **kwargs) -> Dict[str, Any]:
        """Generate file metadata"""
        extensions = ["pdf", "doc", "txt", "csv", "json", "xml", "log"]
        
        return {
            "id": str(uuid.uuid4()),
            "name": f"{self.faker.file_name(extension=random.choice(extensions))}",
            "path": self.faker.file_path(),
            "size": random.randint(1024, 10485760),  # 1KB to 10MB
            "mime_type": self.faker.mime_type(),
            "checksum": hashlib.md5(str(uuid.uuid4()).encode()).hexdigest(),
            "created_at": self.faker.date_time_between("-1y", "now").isoformat(),
            "modified_at": datetime.now().isoformat(),
            "owner": self.faker.user_name(),
            "permissions": random.choice(["644", "755", "777"]),
            "metadata": {
                "tags": [self.faker.word() for _ in range(random.randint(0, 5))],
                "description": self.faker.sentence()
            }
        }
    
    def _generate_image(self, **kwargs) -> Dict[str, Any]:
        """Generate image metadata"""
        return {
            "id": str(uuid.uuid4()),
            "url": self.faker.image_url(),
            "filename": f"{self.faker.file_name(extension='jpg')}",
            "width": random.choice([640, 800, 1024, 1280, 1920]),
            "height": random.choice([480, 600, 768, 720, 1080]),
            "size": random.randint(10240, 5242880),  # 10KB to 5MB
            "format": random.choice(["jpeg", "png", "webp", "gif"]),
            "color_space": random.choice(["rgb", "srgb", "cmyk"]),
            "metadata": {
                "camera": self.faker.company() if random.random() > 0.5 else None,
                "aperture": f"f/{random.choice([1.4, 2.8, 4.0, 5.6, 8.0])}",
                "shutter_speed": f"1/{random.choice([60, 125, 250, 500, 1000])}",
                "iso": random.choice([100, 200, 400, 800, 1600]),
                "location": {
                    "latitude": float(self.faker.latitude()),
                    "longitude": float(self.faker.longitude())
                } if random.random() > 0.5 else None
            }
        }
    
    def _generate_video(self, **kwargs) -> Dict[str, Any]:
        """Generate video metadata"""
        return {
            "id": str(uuid.uuid4()),
            "url": f"https://example.com/videos/{uuid.uuid4()}.mp4",
            "filename": f"{self.faker.file_name(extension='mp4')}",
            "duration": random.randint(10, 3600),  # 10 seconds to 1 hour
            "width": random.choice([1280, 1920, 3840]),
            "height": random.choice([720, 1080, 2160]),
            "framerate": random.choice([24, 25, 30, 60]),
            "bitrate": random.randint(1000000, 10000000),  # 1-10 Mbps
            "codec": random.choice(["h264", "h265", "vp9", "av1"]),
            "format": random.choice(["mp4", "webm", "mkv", "avi"]),
            "size": random.randint(1048576, 1073741824),  # 1MB to 1GB
            "metadata": {
                "title": self.faker.sentence(),
                "description": self.faker.text(max_nb_chars=200),
                "tags": [self.faker.word() for _ in range(random.randint(3, 10))],
                "thumbnail": self.faker.image_url()
            }
        }
    
    def _generate_document(self, **kwargs) -> Dict[str, Any]:
        """Generate document metadata"""
        return {
            "id": str(uuid.uuid4()),
            "title": self.faker.sentence(),
            "filename": self.faker.file_name(extension=random.choice(["pdf", "docx", "xlsx"])),
            "author": self.faker.name(),
            "pages": random.randint(1, 100),
            "words": random.randint(100, 50000),
            "language": random.choice(["en", "es", "fr", "de", "zh"]),
            "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}",
            "status": random.choice(["draft", "review", "approved", "published"]),
            "tags": [self.faker.word() for _ in range(random.randint(2, 8))],
            "created_at": self.faker.date_time_between("-2y", "now").isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {
                "department": random.choice(["HR", "Finance", "IT", "Marketing", "Sales"]),
                "confidentiality": random.choice(["public", "internal", "confidential", "secret"]),
                "retention_date": (datetime.now() + timedelta(days=random.randint(365, 2555))).isoformat()
            }
        }
    
    def _generate_password(self) -> str:
        """Generate secure password"""
        length = random.randint(12, 20)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(random.choice(chars) for _ in range(length))
        
        # Ensure it has at least one of each type
        if not any(c.islower() for c in password):
            password = password[:-1] + random.choice(string.ascii_lowercase)
        if not any(c.isupper() for c in password):
            password = password[:-1] + random.choice(string.ascii_uppercase)
        if not any(c.isdigit() for c in password):
            password = password[:-1] + random.choice(string.digits)
        if not any(c in "!@#$%^&*" for c in password):
            password = password[:-1] + random.choice("!@#$%^&*")
        
        return password
    
    def _generate_api_key(self) -> str:
        """Generate API key"""
        prefix = random.choice(["sk", "pk", "api", "key"])
        key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        return f"{prefix}_{key}"
    
    def _generate_secret(self) -> str:
        """Generate secret key"""
        return base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def _generate_id(self) -> str:
        """Generate various ID formats"""
        id_type = random.choice(["uuid", "numeric", "alphanumeric", "prefixed"])
        
        if id_type == "uuid":
            return str(uuid.uuid4())
        elif id_type == "numeric":
            return str(random.randint(100000, 999999999))
        elif id_type == "alphanumeric":
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        else:  # prefixed
            prefix = random.choice(["usr", "ord", "txn", "doc", "ses"])
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            return f"{prefix}_{suffix}"
    
    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate string from regex pattern"""
        # Simplified pattern generation
        # In production, use a library like 'rstr' for proper regex generation
        if pattern.startswith("^") and pattern.endswith("$"):
            pattern = pattern[1:-1]
        
        # Simple patterns
        if pattern == r"\d+":
            return str(random.randint(0, 999999))
        elif pattern == r"[A-Z]+":
            return ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 10)))
        elif pattern == r"[a-z]+":
            return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        else:
            # Fallback to random string
            return self.faker.lexify("?" * 10)
    
    def _derive_value(self, value: Any, related_value: Any) -> Any:
        """Derive a value based on a related value"""
        # Simple derivation logic
        if isinstance(value, str) and isinstance(related_value, str):
            # Make them related (e.g., same prefix)
            return related_value[:len(related_value)//2] + value[len(value)//2:]
        elif isinstance(value, (int, float)) and isinstance(related_value, (int, float)):
            # Make them proportional
            return related_value * random.uniform(0.8, 1.2)
        else:
            return value


# Example usage
def test_mock_generator():
    """Test the mock data generator"""
    print("\n" + "="*60)
    print("Testing Mock Data Generator")
    print("="*60)
    
    # Create generator
    config = MockConfig(
        locale="en_US",
        realistic=True,
        include_edge_cases=True
    )
    generator = MockDataGenerator(config)
    
    # Test various data types
    test_types = [
        MockDataType.PERSON,
        MockDataType.USER,
        MockDataType.PRODUCT,
        MockDataType.ORDER,
        MockDataType.API_RESPONSE
    ]
    
    for data_type in test_types:
        print(f"\n{data_type.value.upper()}:")
        print("-" * 40)
        data = generator.generate(data_type)
        
        # Show selected fields
        if data_type == MockDataType.PERSON:
            print(f"  Name: {data['first_name']} {data['last_name']}")
            print(f"  Email: {data['email']}")
            print(f"  Phone: {data['phone']}")
        elif data_type == MockDataType.USER:
            print(f"  Username: {data['username']}")
            print(f"  Email: {data['email']}")
            print(f"  Role: {data['role']}")
            print(f"  Status: {data['status']}")
        elif data_type == MockDataType.PRODUCT:
            print(f"  Name: {data['name']}")
            print(f"  Price: ${data['price']}")
            print(f"  Category: {data['category']}")
            print(f"  Stock: {data['stock']}")
        elif data_type == MockDataType.ORDER:
            print(f"  Order #: {data['order_number']}")
            print(f"  Status: {data['status']}")
            print(f"  Total: ${data['total']}")
            print(f"  Items: {len(data['items'])}")
        elif data_type == MockDataType.API_RESPONSE:
            print(f"  Success: {data['success']}")
            print(f"  Status: {data.get('status', data.get('error', {}).get('code'))}")
            if data['success']:
                print(f"  Data ID: {data['data']['id']}")
    
    # Test schema-based generation
    print("\n" + "-"*40)
    print("Schema-Based Generation:")
    
    schema = MockSchema(
        fields={
            "id": "uuid",
            "name": "string",
            "age": "integer",
            "email": "email",
            "is_active": "boolean",
            "created_at": "datetime",
            "tags": "array"
        },
        required_fields=["id", "name", "email"],
        constraints={
            "age": {"min": 18, "max": 65},
            "tags": {"item_type": "string", "min_size": 1, "max_size": 5}
        },
        sample_size=3
    )
    
    schema_data = generator.generate_from_schema(schema)
    print(f"\nGenerated {len(schema_data)} records from schema")
    for i, record in enumerate(schema_data):
        print(f"\nRecord {i+1}:")
        print(f"  ID: {record['id']}")
        print(f"  Name: {record['name']}")
        print(f"  Email: {record['email']}")
        print(f"  Age: {record['age']}")
        print(f"  Tags: {record['tags']}")
    
    return generator


if __name__ == "__main__":
    print("Mock Data Generator for Testing")
    print("="*60)
    
    try:
        generator = test_mock_generator()
        print("\n✅ Mock Data Generator initialized successfully!")
    except ImportError as e:
        print(f"\n⚠️ Warning: {e}")
        print("Install faker with: pip install faker")
        print("Basic mock data generation will still work.")
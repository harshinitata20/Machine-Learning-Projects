"""
Food Expiry Database Management

This module manages the database of food items and their shelf life information.
It provides functionality to store, retrieve, and manage expiry data for different food items.
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging


class ExpiryDatabase:
    """Database manager for food expiry information."""
    
    def __init__(self, db_path: str = "data/food_expiry.db"):
        """
        Initialize the expiry database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        self._populate_default_data()
        
        print(f"✅ Expiry database initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create food_items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    food_name TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    shelf_life_room INTEGER,  -- days at room temperature
                    shelf_life_fridge INTEGER,  -- days in refrigerator
                    shelf_life_freezer INTEGER,  -- days in freezer
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user_items table (for tracking specific items)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    food_name TEXT NOT NULL,
                    purchase_date DATE NOT NULL,
                    expiry_date DATE,
                    storage_location TEXT DEFAULT 'fridge',
                    quantity INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'fresh',  -- fresh, warning, expired
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (food_name) REFERENCES food_items (food_name)
                )
            """)
            
            conn.commit()
    
    def _populate_default_data(self):
        """Populate database with default food expiry data."""
        # Check if data already exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM food_items")
            count = cursor.fetchone()[0]
            
            if count > 0:
                return  # Data already exists
        
        # Default food expiry data
        default_foods = [
            # Fruits
            ("apple", "fruit", 7, 30, 365, "Best stored in fridge for longer life"),
            ("banana", "fruit", 5, 7, 90, "Ripens faster at room temperature"),
            ("orange", "fruit", 7, 14, 365, "Citrus fruits last well in fridge"),
            ("grapes", "fruit", 3, 7, 365, "Store in fridge in perforated bag"),
            ("strawberry", "fruit", 1, 5, 365, "Very perishable, use quickly"),
            
            # Vegetables
            ("broccoli", "vegetable", 2, 7, 365, "Store in fridge, use quickly"),
            ("carrot", "vegetable", 7, 30, 365, "Remove green tops before storing"),
            ("lettuce", "vegetable", 1, 7, 90, "Store in crisper drawer"),
            ("tomato", "vegetable", 5, 7, 90, "Better flavor at room temperature"),
            ("onion", "vegetable", 30, 60, 365, "Store in cool, dry place"),
            
            # Dairy
            ("milk", "dairy", 1, 7, 90, "Check sell-by date"),
            ("cheese", "dairy", 7, 30, 180, "Hard cheeses last longer"),
            ("yogurt", "dairy", 1, 14, 60, "Check expiration date"),
            ("butter", "dairy", 7, 60, 365, "Can be frozen for longer storage"),
            ("eggs", "dairy", 14, 30, 365, "Store in original carton"),
            
            # Proteins
            ("chicken", "protein", 1, 3, 365, "Cook within 1-2 days of purchase"),
            ("beef", "protein", 2, 5, 365, "Ground beef expires faster"),
            ("fish", "protein", 1, 2, 180, "Very perishable, use quickly"),
            ("tofu", "protein", 1, 7, 180, "Change water daily if opened"),
            
            # Grains & Bakery
            ("bread", "grain", 5, 7, 90, "Freeze for longer storage"),
            ("rice", "grain", 365, 365, 730, "Dry storage in airtight container"),
            ("pasta", "grain", 730, 730, 730, "Dry pasta lasts very long"),
            ("cereal", "grain", 365, 365, 365, "Keep in airtight container"),
            
            # Processed Foods
            ("pizza", "processed", 1, 3, 60, "Leftover pizza"),
            ("sandwich", "processed", 1, 2, 30, "Depends on ingredients"),
            ("cake", "processed", 2, 7, 90, "Depends on frosting type"),
            ("donut", "processed", 2, 5, 30, "Best consumed fresh")
        ]
        
        # Insert default data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO food_items 
                (food_name, category, shelf_life_room, shelf_life_fridge, shelf_life_freezer, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, default_foods)
            conn.commit()
        
        print(f"📚 Populated database with {len(default_foods)} default food items")
    
    def get_shelf_life(self, 
                      food_name: str, 
                      storage_location: str = "fridge") -> Optional[int]:
        """
        Get shelf life for a food item based on storage location.
        
        Args:
            food_name: Name of the food item
            storage_location: 'room', 'fridge', or 'freezer'
            
        Returns:
            Shelf life in days, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute("""
                SELECT shelf_life_room, shelf_life_fridge, shelf_life_freezer 
                FROM food_items WHERE LOWER(food_name) = LOWER(?)
            """, (food_name,))
            
            result = cursor.fetchone()
            
            if not result:
                # Try partial match
                cursor.execute("""
                    SELECT shelf_life_room, shelf_life_fridge, shelf_life_freezer 
                    FROM food_items WHERE LOWER(food_name) LIKE LOWER(?)
                """, (f"%{food_name}%",))
                result = cursor.fetchone()
            
            if result:
                room_days, fridge_days, freezer_days = result
                
                if storage_location.lower() == "room":
                    return room_days
                elif storage_location.lower() == "fridge":
                    return fridge_days
                elif storage_location.lower() == "freezer":
                    return freezer_days
        
        return None
    
    def calculate_expiry_date(self, 
                            food_name: str,
                            purchase_date: str,
                            storage_location: str = "fridge") -> Optional[str]:
        """
        Calculate expiry date for a food item.
        
        Args:
            food_name: Name of the food item
            purchase_date: Purchase date in YYYY-MM-DD format
            storage_location: Storage location
            
        Returns:
            Expiry date in YYYY-MM-DD format, or None if not found
        """
        shelf_life = self.get_shelf_life(food_name, storage_location)
        
        if shelf_life:
            purchase_dt = datetime.strptime(purchase_date, "%Y-%m-%d")
            expiry_dt = purchase_dt + timedelta(days=shelf_life)
            return expiry_dt.strftime("%Y-%m-%d")
        
        return None
    
    def add_user_item(self, 
                     food_name: str,
                     purchase_date: str,
                     storage_location: str = "fridge",
                     quantity: int = 1) -> int:
        """
        Add a user's food item to tracking.
        
        Args:
            food_name: Name of the food item
            purchase_date: Purchase date in YYYY-MM-DD format
            storage_location: Storage location
            quantity: Quantity of items
            
        Returns:
            ID of the inserted item
        """
        # Calculate expiry date
        expiry_date = self.calculate_expiry_date(food_name, purchase_date, storage_location)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_items 
                (food_name, purchase_date, expiry_date, storage_location, quantity)
                VALUES (?, ?, ?, ?, ?)
            """, (food_name, purchase_date, expiry_date, storage_location, quantity))
            
            item_id = cursor.lastrowid
            conn.commit()
        
        return item_id
    
    def get_expiring_items(self, days_ahead: int = 3) -> List[Dict]:
        """
        Get items that are expiring within the specified number of days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of expiring items with details
        """
        cutoff_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.food_name, u.purchase_date, u.expiry_date,
                       u.storage_location, u.quantity, u.status,
                       f.category, f.notes,
                       julianday(u.expiry_date) - julianday('now') as days_remaining
                FROM user_items u
                LEFT JOIN food_items f ON u.food_name = f.food_name
                WHERE u.expiry_date <= ? AND u.status != 'consumed'
                ORDER BY u.expiry_date ASC
            """, (cutoff_date,))
            
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return results
    
    def update_item_status(self, item_id: int, status: str):
        """
        Update the status of a user item.
        
        Args:
            item_id: ID of the item to update
            status: New status ('fresh', 'warning', 'expired', 'consumed')
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_items 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, item_id))
            conn.commit()
    
    def get_food_categories(self) -> List[str]:
        """Get all food categories in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT category FROM food_items ORDER BY category")
            return [row[0] for row in cursor.fetchall()]
    
    def search_foods(self, query: str) -> List[Dict]:
        """
        Search for food items by name.
        
        Args:
            query: Search query
            
        Returns:
            List of matching food items
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT food_name, category, shelf_life_room, shelf_life_fridge, 
                       shelf_life_freezer, notes
                FROM food_items 
                WHERE LOWER(food_name) LIKE LOWER(?)
                ORDER BY food_name
            """, (f"%{query}%",))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total foods in database
            cursor.execute("SELECT COUNT(*) FROM food_items")
            total_foods = cursor.fetchone()[0]
            
            # Total user items
            cursor.execute("SELECT COUNT(*) FROM user_items")
            total_user_items = cursor.fetchone()[0]
            
            # Items by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM user_items 
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Items by category
            cursor.execute("""
                SELECT f.category, COUNT(*) 
                FROM user_items u
                JOIN food_items f ON u.food_name = f.food_name
                GROUP BY f.category
            """)
            category_counts = dict(cursor.fetchall())
        
        return {
            "total_foods_in_db": total_foods,
            "total_user_items": total_user_items,
            "items_by_status": status_counts,
            "items_by_category": category_counts
        }
    
    def export_to_json(self, output_path: str):
        """Export database to JSON format."""
        with sqlite3.connect(self.db_path) as conn:
            # Export food items
            food_items_df = pd.read_sql_query("SELECT * FROM food_items", conn)
            user_items_df = pd.read_sql_query("SELECT * FROM user_items", conn)
            
            export_data = {
                "food_items": food_items_df.to_dict(orient="records"),
                "user_items": user_items_df.to_dict(orient="records"),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"📤 Database exported to {output_path}")


# Dictionary-based alternative for simple usage
FOOD_EXPIRY_DICT = {
    "apple": {"fridge": 30, "room": 7, "freezer": 365},
    "banana": {"fridge": 7, "room": 5, "freezer": 90},
    "bread": {"fridge": 7, "room": 5, "freezer": 90},
    "milk": {"fridge": 7, "room": 1, "freezer": 90},
    "cheese": {"fridge": 30, "room": 7, "freezer": 180},
    "eggs": {"fridge": 30, "room": 14, "freezer": 365},
    "chicken": {"fridge": 3, "room": 1, "freezer": 365},
    "broccoli": {"fridge": 7, "room": 2, "freezer": 365},
    "tomato": {"fridge": 7, "room": 5, "freezer": 90},
    "orange": {"fridge": 14, "room": 7, "freezer": 365},
    "lettuce": {"fridge": 7, "room": 1, "freezer": 90},
    "carrot": {"fridge": 30, "room": 7, "freezer": 365},
    "onion": {"fridge": 60, "room": 30, "freezer": 365},
    "pizza": {"fridge": 3, "room": 1, "freezer": 60},
    "sandwich": {"fridge": 2, "room": 1, "freezer": 30}
}


def get_expiry_simple(food_name: str, storage: str = "fridge") -> int:
    """
    Simple function to get expiry days using dictionary.
    
    Args:
        food_name: Name of the food
        storage: Storage location
        
    Returns:
        Number of days until expiry
    """
    food_name = food_name.lower()
    
    if food_name in FOOD_EXPIRY_DICT:
        return FOOD_EXPIRY_DICT[food_name].get(storage, 7)  # Default to 7 days
    
    # Try partial matching
    for food in FOOD_EXPIRY_DICT:
        if food in food_name or food_name in food:
            return FOOD_EXPIRY_DICT[food].get(storage, 7)
    
    return 7  # Default fallback


if __name__ == "__main__":
    # Demo the expiry database
    print("🗄️ Food Expiry Database Demo")
    print("=" * 50)
    
    # Initialize database
    db = ExpiryDatabase()
    
    # Add some sample user items
    sample_items = [
        ("apple", "2025-09-28", "fridge", 3),
        ("milk", "2025-09-30", "fridge", 1),
        ("bread", "2025-10-01", "room", 1),
        ("chicken", "2025-09-29", "fridge", 2)
    ]
    
    for item in sample_items:
        item_id = db.add_user_item(*item)
        print(f"➕ Added item {item[0]} with ID {item_id}")
    
    # Check expiring items
    expiring = db.get_expiring_items(days_ahead=5)
    print(f"\n⚠️ Items expiring in next 5 days: {len(expiring)}")
    for item in expiring:
        days_left = int(item['days_remaining']) if item['days_remaining'] else 0
        print(f"  • {item['food_name']}: {days_left} days remaining")
    
    # Show statistics
    stats = db.get_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"  • Total foods in database: {stats['total_foods_in_db']}")
    print(f"  • Total user items: {stats['total_user_items']}")
    
    print("\n✅ Expiry database system ready!")
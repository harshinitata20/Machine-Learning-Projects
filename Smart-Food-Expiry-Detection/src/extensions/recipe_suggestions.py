"""
Recipe Suggestions Extension

This module provides AI-powered recipe suggestions for items nearing expiry.
Integrates with external recipe APIs and uses local knowledge to suggest recipes.
"""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import random


class RecipeSuggester:
    """Recipe suggestion engine for expiring food items."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize recipe suggester.
        
        Args:
            api_key: Optional API key for external recipe services
        """
        self.api_key = api_key
        self.base_recipes = self._load_base_recipes()
        
        # External API configurations
        self.apis = {
            "spoonacular": {
                "base_url": "https://api.spoonacular.com/recipes",
                "key_param": "apiKey"
            },
            "edamam": {
                "base_url": "https://api.edamam.com/search",
                "key_param": "app_key"
            }
        }
    
    def _load_base_recipes(self) -> Dict:
        """Load base recipe database."""
        return {
            "quick_stir_fry": {
                "name": "Quick Vegetable Stir Fry",
                "ingredients": ["vegetables", "oil", "garlic", "soy_sauce"],
                "time": "15 minutes",
                "difficulty": "Easy",
                "instructions": [
                    "Heat oil in a large pan or wok",
                    "Add minced garlic and cook for 30 seconds",
                    "Add vegetables and stir-fry for 5-8 minutes", 
                    "Add soy sauce and cook for 2 more minutes",
                    "Serve immediately over rice or noodles"
                ],
                "categories": ["vegetables", "quick", "healthy"]
            },
            
            "fresh_salad": {
                "name": "Fresh Garden Salad",
                "ingredients": ["lettuce", "tomato", "cucumber", "dressing"],
                "time": "10 minutes",
                "difficulty": "Easy",
                "instructions": [
                    "Wash and chop all vegetables",
                    "Combine in a large bowl",
                    "Add dressing and toss to combine",
                    "Serve immediately"
                ],
                "categories": ["vegetables", "raw", "healthy", "quick"]
            },
            
            "fruit_smoothie": {
                "name": "Mixed Fruit Smoothie",
                "ingredients": ["banana", "apple", "milk", "honey"],
                "time": "5 minutes", 
                "difficulty": "Easy",
                "instructions": [
                    "Peel and chop fruits",
                    "Add all ingredients to blender",
                    "Blend until smooth",
                    "Serve in glasses with ice"
                ],
                "categories": ["fruits", "drinks", "healthy", "quick"]
            },
            
            "egg_scramble": {
                "name": "Vegetable Egg Scramble",
                "ingredients": ["eggs", "vegetables", "cheese", "butter"],
                "time": "10 minutes",
                "difficulty": "Easy", 
                "instructions": [
                    "Beat eggs in a bowl",
                    "Heat butter in a pan",
                    "Add vegetables and cook for 3 minutes",
                    "Add eggs and scramble until cooked",
                    "Top with cheese and serve"
                ],
                "categories": ["eggs", "breakfast", "protein", "quick"]
            },
            
            "chicken_soup": {
                "name": "Simple Chicken Soup",
                "ingredients": ["chicken", "vegetables", "broth", "herbs"],
                "time": "45 minutes",
                "difficulty": "Medium",
                "instructions": [
                    "Cut chicken into pieces",
                    "Sauté vegetables in pot",
                    "Add chicken and broth",
                    "Simmer for 30 minutes",
                    "Season with herbs and serve"
                ],
                "categories": ["chicken", "soup", "comfort", "protein"]
            }
        }
    
    def get_suggestions(self, 
                       food_items: List[str],
                       dietary_restrictions: List[str] = None,
                       max_time: int = 60,
                       difficulty: str = "any") -> List[Dict]:
        """
        Get recipe suggestions for given food items.
        
        Args:
            food_items: List of food items to use
            dietary_restrictions: List of dietary restrictions
            max_time: Maximum cooking time in minutes
            difficulty: Preferred difficulty level
            
        Returns:
            List of recipe suggestions
        """
        suggestions = []
        
        # Try external API first (if available)
        if self.api_key:
            external_recipes = self._get_external_recipes(
                food_items, dietary_restrictions, max_time
            )
            suggestions.extend(external_recipes)
        
        # Add local recipe suggestions
        local_recipes = self._get_local_recipes(
            food_items, dietary_restrictions, max_time, difficulty
        )
        suggestions.extend(local_recipes)
        
        # Score and sort recipes
        scored_recipes = self._score_recipes(suggestions, food_items)
        
        # Return top suggestions
        return sorted(scored_recipes, key=lambda x: x["score"], reverse=True)[:10]
    
    def _get_external_recipes(self, 
                            food_items: List[str],
                            dietary_restrictions: List[str] = None,
                            max_time: int = 60) -> List[Dict]:
        """Get recipes from external API."""
        recipes = []
        
        try:
            # Example with Spoonacular API (requires API key)
            if "spoonacular" in self.apis:
                params = {
                    "ingredients": ",".join(food_items),
                    "number": 5,
                    "maxReadyTime": max_time,
                    self.apis["spoonacular"]["key_param"]: self.api_key
                }
                
                if dietary_restrictions:
                    params["diet"] = ",".join(dietary_restrictions)
                
                url = f"{self.apis['spoonacular']['base_url']}/findByIngredients"
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    api_recipes = response.json()
                    
                    for recipe in api_recipes:
                        recipes.append({
                            "name": recipe.get("title", "Unknown Recipe"),
                            "ingredients": [ing["name"] for ing in recipe.get("usedIngredients", [])],
                            "time": f"{recipe.get('readyInMinutes', 'N/A')} minutes",
                            "difficulty": "Medium",  # Default as API doesn't provide
                            "url": f"https://spoonacular.com/recipes/{recipe.get('id')}",
                            "source": "Spoonacular",
                            "image": recipe.get("image"),
                            "missing_ingredients": len(recipe.get("missedIngredients", [])),
                            "used_ingredients": len(recipe.get("usedIngredients", []))
                        })
        
        except Exception as e:
            print(f"Error fetching external recipes: {e}")
        
        return recipes
    
    def _get_local_recipes(self, 
                          food_items: List[str],
                          dietary_restrictions: List[str] = None,
                          max_time: int = 60,
                          difficulty: str = "any") -> List[Dict]:
        """Get recipes from local database."""
        suggestions = []
        
        # Normalize food items
        normalized_items = [item.lower().strip() for item in food_items]
        
        for recipe_id, recipe in self.base_recipes.items():
            # Check if recipe uses any of the input ingredients
            recipe_ingredients = [ing.lower() for ing in recipe["ingredients"]]
            
            matching_ingredients = []
            for item in normalized_items:
                for recipe_ing in recipe_ingredients:
                    if item in recipe_ing or recipe_ing in item:
                        matching_ingredients.append(item)
            
            if matching_ingredients:
                # Check time constraint
                recipe_time = self._extract_time_minutes(recipe["time"])
                if recipe_time and recipe_time > max_time:
                    continue
                
                # Check difficulty
                if difficulty != "any" and recipe["difficulty"].lower() != difficulty.lower():
                    continue
                
                # Check dietary restrictions
                if dietary_restrictions and not self._check_dietary_compatibility(
                    recipe, dietary_restrictions
                ):
                    continue
                
                suggestions.append({
                    "name": recipe["name"],
                    "ingredients": recipe["ingredients"],
                    "time": recipe["time"],
                    "difficulty": recipe["difficulty"],
                    "instructions": recipe["instructions"],
                    "categories": recipe["categories"],
                    "source": "Local Database",
                    "matching_ingredients": matching_ingredients,
                    "id": recipe_id
                })
        
        return suggestions
    
    def _extract_time_minutes(self, time_str: str) -> Optional[int]:
        """Extract time in minutes from time string."""
        try:
            import re
            match = re.search(r'(\d+)', time_str)
            return int(match.group(1)) if match else None
        except:
            return None
    
    def _check_dietary_compatibility(self, 
                                   recipe: Dict,
                                   restrictions: List[str]) -> bool:
        """Check if recipe is compatible with dietary restrictions."""
        # Simple compatibility check based on ingredients and categories
        restriction_keywords = {
            "vegetarian": ["meat", "chicken", "beef", "fish", "pork"],
            "vegan": ["meat", "chicken", "beef", "fish", "pork", "dairy", "eggs", "cheese", "milk"],
            "gluten-free": ["bread", "pasta", "wheat", "flour"],
            "dairy-free": ["milk", "cheese", "butter", "cream", "yogurt"]
        }
        
        recipe_text = " ".join(recipe["ingredients"] + recipe["categories"]).lower()
        
        for restriction in restrictions:
            if restriction.lower() in restriction_keywords:
                forbidden_items = restriction_keywords[restriction.lower()]
                if any(item in recipe_text for item in forbidden_items):
                    return False
        
        return True
    
    def _score_recipes(self, recipes: List[Dict], input_items: List[str]) -> List[Dict]:
        """Score recipes based on ingredient matching and other factors."""
        for recipe in recipes:
            score = 0
            
            # Score based on ingredient matching
            if "matching_ingredients" in recipe:
                matching_count = len(recipe["matching_ingredients"])
                score += matching_count * 10
            elif "used_ingredients" in recipe:
                score += recipe["used_ingredients"] * 10
            
            # Bonus for fewer missing ingredients
            if "missing_ingredients" in recipe:
                score -= recipe["missing_ingredients"] * 2
            
            # Bonus for quick recipes
            recipe_time = self._extract_time_minutes(recipe["time"])
            if recipe_time:
                if recipe_time <= 15:
                    score += 5
                elif recipe_time <= 30:
                    score += 2
            
            # Bonus for easy recipes
            if recipe["difficulty"].lower() == "easy":
                score += 3
            
            recipe["score"] = score
        
        return recipes
    
    def generate_smart_suggestions(self, expiring_items: List[Dict]) -> List[Dict]:
        """
        Generate smart recipe suggestions based on expiring items.
        
        Args:
            expiring_items: List of items with expiry information
            
        Returns:
            Prioritized recipe suggestions
        """
        # Group items by urgency
        urgent_items = []  # Expire in 1 day
        soon_items = []    # Expire in 2-3 days
        
        for item in expiring_items:
            days_remaining = item.get("days_remaining", 0)
            
            if days_remaining <= 1:
                urgent_items.append(item["food_name"])
            elif days_remaining <= 3:
                soon_items.append(item["food_name"])
        
        suggestions = []
        
        # Priority 1: Recipes using urgent items
        if urgent_items:
            urgent_recipes = self.get_suggestions(
                urgent_items, 
                max_time=30,  # Quick recipes for urgent items
                difficulty="easy"
            )
            
            for recipe in urgent_recipes:
                recipe["priority"] = "urgent"
                recipe["reason"] = f"Uses items expiring today: {', '.join(urgent_items)}"
            
            suggestions.extend(urgent_recipes[:3])
        
        # Priority 2: Recipes using soon-expiring items
        if soon_items:
            soon_recipes = self.get_suggestions(
                soon_items,
                max_time=60,
                difficulty="any"
            )
            
            for recipe in soon_recipes:
                recipe["priority"] = "moderate"
                recipe["reason"] = f"Uses items expiring soon: {', '.join(soon_items)}"
            
            suggestions.extend(soon_recipes[:3])
        
        # Priority 3: Combo recipes using multiple expiring items
        all_expiring = [item["food_name"] for item in expiring_items]
        if len(all_expiring) > 1:
            combo_recipes = self.get_suggestions(
                all_expiring,
                max_time=45,
                difficulty="any"
            )
            
            for recipe in combo_recipes:
                recipe["priority"] = "combo"
                recipe["reason"] = "Uses multiple expiring ingredients"
            
            suggestions.extend(combo_recipes[:2])
        
        return suggestions
    
    def format_recipe_for_display(self, recipe: Dict) -> str:
        """Format recipe for display in notifications or UI."""
        lines = [
            f"🍳 **{recipe['name']}**",
            f"⏰ Time: {recipe['time']}",
            f"📊 Difficulty: {recipe['difficulty']}"
        ]
        
        if "reason" in recipe:
            lines.append(f"💡 Why: {recipe['reason']}")
        
        if "ingredients" in recipe:
            lines.append(f"🥘 Key ingredients: {', '.join(recipe['ingredients'][:5])}")
        
        if "url" in recipe:
            lines.append(f"🔗 Full recipe: {recipe['url']}")
        
        return "\n".join(lines)


def demo_recipe_suggestions():
    """Demonstration of recipe suggestion functionality."""
    print("🍳 Recipe Suggestions Demo")
    print("=" * 50)
    
    suggester = RecipeSuggester()
    
    # Demo with expiring items
    expiring_items = [
        {"food_name": "apple", "days_remaining": 1},
        {"food_name": "banana", "days_remaining": 2},
        {"food_name": "lettuce", "days_remaining": 1},
        {"food_name": "eggs", "days_remaining": 3}
    ]
    
    print("🚨 Expiring Items:")
    for item in expiring_items:
        print(f"  • {item['food_name']}: {item['days_remaining']} days")
    
    print("\n🍽️ Smart Recipe Suggestions:")
    
    suggestions = suggester.generate_smart_suggestions(expiring_items)
    
    for i, recipe in enumerate(suggestions[:5], 1):
        print(f"\n{i}. {recipe['name']}")
        print(f"   Priority: {recipe.get('priority', 'normal').title()}")
        print(f"   Time: {recipe['time']}")
        print(f"   Reason: {recipe.get('reason', 'Good match for ingredients')}")
    
    print("\n✅ Recipe suggestion system ready!")
    return suggester


if __name__ == "__main__":
    # Run demo
    suggester = demo_recipe_suggestions()
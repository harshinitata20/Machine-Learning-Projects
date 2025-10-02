"""
Notification Extensions

This module provides various notification channels including email, SMS, WhatsApp, and Telegram
for sending expiry alerts and reminders to users.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Optional, Any
import os
import json
from datetime import datetime
import requests


class EmailNotifier:
    """Email notification service."""
    
    def __init__(self, 
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 username: str = None,
                 password: str = None):
        """
        Initialize email notifier.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: Email username
            password: Email password or app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username or os.getenv("EMAIL_USERNAME")
        self.password = password or os.getenv("EMAIL_PASSWORD")
    
    def send_expiry_alert(self, 
                         recipient: str,
                         expiring_items: List[Dict],
                         include_recipes: bool = True) -> Dict:
        """
        Send expiry alert email.
        
        Args:
            recipient: Recipient email address
            expiring_items: List of expiring items
            include_recipes: Whether to include recipe suggestions
            
        Returns:
            Status dictionary
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 Food Expiry Alert - {len(expiring_items)} items need attention"
            msg['From'] = self.username
            msg['To'] = recipient
            
            # Create HTML content
            html_content = self._create_expiry_email_html(expiring_items, include_recipes)
            
            # Create plain text version
            text_content = self._create_expiry_email_text(expiring_items)
            
            # Attach parts
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.username, recipient, text)
            server.quit()
            
            return {
                "success": True,
                "message": f"Expiry alert sent to {recipient}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_expiry_email_html(self, expiring_items: List[Dict], include_recipes: bool) -> str:
        """Create HTML email content."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .item {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                .urgent {{ border-left-color: #dc3545; }}
                .warning {{ border-left-color: #ffc107; }}
                .fresh {{ border-left-color: #28a745; }}
                .recipe {{ background-color: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .footer {{ margin-top: 30px; padding: 15px; background-color: #f8f9fa; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🍎 Smart Food Expiry Alert</h2>
                <p>You have {len(expiring_items)} food items that need attention!</p>
                <p><strong>Alert Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h3>📋 Items Requiring Attention</h3>
        """
        
        for item in expiring_items:
            days_remaining = item.get('days_remaining', 0)
            
            if days_remaining <= 0:
                css_class = "urgent"
                status = "⛔ EXPIRED"
            elif days_remaining <= 1:
                css_class = "urgent" 
                status = "🔴 Expires Today"
            elif days_remaining <= 3:
                css_class = "warning"
                status = f"🟡 {days_remaining} days left"
            else:
                css_class = "fresh"
                status = f"🟢 {days_remaining} days left"
            
            html += f"""
            <div class="item {css_class}">
                <h4>{item['food_name'].title()}</h4>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Purchase Date:</strong> {item.get('purchase_date', 'Unknown')}</p>
                <p><strong>Storage:</strong> {item.get('storage_location', 'Unknown')}</p>
            </div>
            """
        
        # Add recipe suggestions if requested
        if include_recipes:
            html += """
            <h3>🍳 Suggested Recipes</h3>
            <div class="recipe">
                <h4>Quick Stir Fry</h4>
                <p>Use your expiring vegetables in a delicious stir fry!</p>
                <p><strong>Time:</strong> 15 minutes | <strong>Difficulty:</strong> Easy</p>
            </div>
            <div class="recipe">
                <h4>Fresh Fruit Smoothie</h4>
                <p>Blend expiring fruits into a nutritious smoothie.</p>
                <p><strong>Time:</strong> 5 minutes | <strong>Difficulty:</strong> Easy</p>
            </div>
            """
        
        html += """
            <div class="footer">
                <p>This alert was sent by your Smart Food Expiry Detection System.</p>
                <p>Stay fresh, reduce waste! 🌱</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_expiry_email_text(self, expiring_items: List[Dict]) -> str:
        """Create plain text email content."""
        lines = [
            "🍎 SMART FOOD EXPIRY ALERT",
            "=" * 40,
            f"Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Items requiring attention: {len(expiring_items)}",
            "",
            "📋 EXPIRING ITEMS:"
        ]
        
        for item in expiring_items:
            days_remaining = item.get('days_remaining', 0)
            
            if days_remaining <= 0:
                status = "⛔ EXPIRED"
            elif days_remaining <= 1:
                status = "🔴 Expires Today"
            elif days_remaining <= 3:
                status = f"🟡 {days_remaining} days left"
            else:
                status = f"🟢 {days_remaining} days left"
            
            lines.extend([
                f"  • {item['food_name'].title()}",
                f"    Status: {status}",
                f"    Purchased: {item.get('purchase_date', 'Unknown')}",
                f"    Storage: {item.get('storage_location', 'Unknown')}",
                ""
            ])
        
        lines.extend([
            "💡 TIPS:",
            "  • Use expiring items in stir-fries or smoothies",
            "  • Check your app for personalized recipe suggestions", 
            "  • Consider freezing items to extend shelf life",
            "",
            "Stay fresh, reduce waste! 🌱",
            "- Your Smart Food Expiry Detection System"
        ])
        
        return "\n".join(lines)


class SMSNotifier:
    """SMS notification service using Twilio."""
    
    def __init__(self, account_sid: str = None, auth_token: str = None, from_number: str = None):
        """
        Initialize SMS notifier.
        
        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            from_number: Twilio phone number
        """
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_PHONE_NUMBER")
    
    def send_expiry_alert(self, recipient: str, expiring_items: List[Dict]) -> Dict:
        """
        Send SMS expiry alert.
        
        Args:
            recipient: Recipient phone number
            expiring_items: List of expiring items
            
        Returns:
            Status dictionary
        """
        try:
            # Import Twilio client (optional dependency)
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            # Create message content
            message_body = self._create_sms_content(expiring_items)
            
            # Send SMS
            message = client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=recipient
            )
            
            return {
                "success": True,
                "message": f"SMS alert sent to {recipient}",
                "message_sid": message.sid,
                "timestamp": datetime.now().isoformat()
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "Twilio library not installed. Install with: pip install twilio",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_sms_content(self, expiring_items: List[Dict]) -> str:
        """Create SMS message content."""
        if not expiring_items:
            return "🍎 No items expiring soon. Great job managing your food!"
        
        urgent_count = len([item for item in expiring_items if item.get('days_remaining', 0) <= 1])
        
        lines = [
            f"🚨 FOOD ALERT: {len(expiring_items)} items need attention!"
        ]
        
        if urgent_count > 0:
            lines.append(f"⛔ {urgent_count} expire today/expired")
        
        # Add top 3 most urgent items
        sorted_items = sorted(expiring_items, key=lambda x: x.get('days_remaining', 0))
        
        for item in sorted_items[:3]:
            days = item.get('days_remaining', 0)
            if days <= 0:
                status = "EXPIRED"
            elif days <= 1:
                status = "TODAY"
            else:
                status = f"{days}d"
            
            lines.append(f"• {item['food_name']}: {status}")
        
        if len(expiring_items) > 3:
            lines.append(f"+ {len(expiring_items) - 3} more items")
        
        lines.append("Check your app for recipes! 🍳")
        
        return "\n".join(lines)


class WhatsAppNotifier:
    """WhatsApp notification service."""
    
    def __init__(self, api_token: str = None):
        """
        Initialize WhatsApp notifier.
        
        Args:
            api_token: WhatsApp Business API token
        """
        self.api_token = api_token or os.getenv("WHATSAPP_API_TOKEN")
        self.base_url = "https://graph.facebook.com/v17.0"
    
    def send_expiry_alert(self, recipient: str, expiring_items: List[Dict]) -> Dict:
        """
        Send WhatsApp expiry alert.
        
        Args:
            recipient: Recipient WhatsApp number
            expiring_items: List of expiring items
            
        Returns:
            Status dictionary
        """
        try:
            # Create message content
            message_text = self._create_whatsapp_content(expiring_items)
            
            # WhatsApp Business API endpoint
            phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
            url = f"{self.base_url}/{phone_number_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": recipient,
                "type": "text",
                "text": {"body": message_text}
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"WhatsApp alert sent to {recipient}",
                    "response": response.json(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_whatsapp_content(self, expiring_items: List[Dict]) -> str:
        """Create WhatsApp message content."""
        lines = [
            "🍎 *Food Expiry Alert*",
            "",
            f"You have *{len(expiring_items)}* items that need attention:"
        ]
        
        for item in expiring_items:
            days = item.get('days_remaining', 0)
            
            if days <= 0:
                emoji = "⛔"
                status = "EXPIRED"
            elif days <= 1:
                emoji = "🔴"
                status = "Expires today"
            elif days <= 3:
                emoji = "🟡"
                status = f"{days} days left"
            else:
                emoji = "🟢"
                status = f"{days} days left"
            
            lines.append(f"{emoji} *{item['food_name'].title()}*: {status}")
        
        lines.extend([
            "",
            "💡 *Quick suggestions:*",
            "• Make a stir-fry with expiring vegetables",
            "• Blend fruits into smoothies", 
            "• Check your app for personalized recipes",
            "",
            "Stay fresh, reduce waste! 🌱"
        ])
        
        return "\n".join(lines)


class TelegramNotifier:
    """Telegram notification service."""
    
    def __init__(self, bot_token: str = None):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_expiry_alert(self, chat_id: str, expiring_items: List[Dict]) -> Dict:
        """
        Send Telegram expiry alert.
        
        Args:
            chat_id: Telegram chat ID
            expiring_items: List of expiring items
            
        Returns:
            Status dictionary
        """
        try:
            message_text = self._create_telegram_content(expiring_items)
            
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                "chat_id": chat_id,
                "text": message_text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Telegram alert sent to chat {chat_id}",
                    "response": response.json(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "response": response.text,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_telegram_content(self, expiring_items: List[Dict]) -> str:
        """Create Telegram message content."""
        lines = [
            "🍎 *Food Expiry Alert*",
            "",
            f"You have *{len(expiring_items)}* items that need attention:",
            ""
        ]
        
        for item in expiring_items:
            days = item.get('days_remaining', 0)
            
            if days <= 0:
                emoji = "⛔"
                status = "EXPIRED"
            elif days <= 1:
                emoji = "🔴"
                status = "Expires today"
            elif days <= 3:
                emoji = "🟡"
                status = f"{days} days left"
            else:
                emoji = "🟢"
                status = f"{days} days left"
            
            lines.append(f"{emoji} *{item['food_name'].title()}*: {status}")
            lines.append(f"   📅 Purchased: {item.get('purchase_date', 'Unknown')}")
            lines.append("")
        
        lines.extend([
            "💡 *Quick Tips:*",
            "• Use expiring veggies in stir-fries",
            "• Make smoothies with aging fruits",
            "• Check your app for recipe ideas",
            "",
            "Stay fresh! 🌱"
        ])
        
        return "\n".join(lines)


class NotificationManager:
    """Centralized notification management."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.email_notifier = EmailNotifier()
        self.sms_notifier = SMSNotifier()
        self.whatsapp_notifier = WhatsAppNotifier()
        self.telegram_notifier = TelegramNotifier()
    
    def send_expiry_alerts(self, 
                          recipients: Dict[str, str],
                          expiring_items: List[Dict],
                          channels: List[str] = ["email"]) -> Dict:
        """
        Send expiry alerts through multiple channels.
        
        Args:
            recipients: Dictionary of channel -> recipient info
            expiring_items: List of expiring items
            channels: List of notification channels to use
            
        Returns:
            Summary of notification results
        """
        results = {}
        
        for channel in channels:
            if channel == "email" and "email" in recipients:
                results["email"] = self.email_notifier.send_expiry_alert(
                    recipients["email"], expiring_items
                )
            
            elif channel == "sms" and "sms" in recipients:
                results["sms"] = self.sms_notifier.send_expiry_alert(
                    recipients["sms"], expiring_items
                )
            
            elif channel == "whatsapp" and "whatsapp" in recipients:
                results["whatsapp"] = self.whatsapp_notifier.send_expiry_alert(
                    recipients["whatsapp"], expiring_items
                )
            
            elif channel == "telegram" and "telegram" in recipients:
                results["telegram"] = self.telegram_notifier.send_expiry_alert(
                    recipients["telegram"], expiring_items
                )
        
        # Calculate summary
        successful = sum(1 for result in results.values() if result.get("success"))
        failed = len(results) - successful
        
        return {
            "summary": {
                "total_sent": len(results),
                "successful": successful,
                "failed": failed
            },
            "details": results,
            "timestamp": datetime.now().isoformat()
        }


def demo_notifications():
    """Demonstration of notification functionality."""
    print("🔔 Notification System Demo")
    print("=" * 50)
    
    # Sample expiring items
    expiring_items = [
        {
            "food_name": "milk",
            "days_remaining": 1,
            "purchase_date": "2025-09-30",
            "storage_location": "fridge"
        },
        {
            "food_name": "bread", 
            "days_remaining": 0,
            "purchase_date": "2025-09-26",
            "storage_location": "room"
        },
        {
            "food_name": "apples",
            "days_remaining": 2,
            "purchase_date": "2025-09-28",
            "storage_location": "fridge"
        }
    ]
    
    print("📋 Sample expiring items:")
    for item in expiring_items:
        print(f"  • {item['food_name']}: {item['days_remaining']} days remaining")
    
    # Initialize notification manager
    notification_manager = NotificationManager()
    
    print("\n📧 Email notification content preview:")
    email_notifier = EmailNotifier()
    text_content = email_notifier._create_expiry_email_text(expiring_items)
    print(text_content[:500] + "..." if len(text_content) > 500 else text_content)
    
    print("\n📱 SMS notification content preview:")
    sms_notifier = SMSNotifier()
    sms_content = sms_notifier._create_sms_content(expiring_items)
    print(sms_content)
    
    print("\n✅ Notification system ready!")
    print("Note: Configure API keys in environment variables to enable sending.")
    
    return notification_manager


if __name__ == "__main__":
    # Run demo
    notification_manager = demo_notifications()
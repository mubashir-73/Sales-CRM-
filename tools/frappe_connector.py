import os
from typing import Dict, Optional

import requests


class FrappeConnector:
    def __init__(self):
        self.site_url = os.getenv("FRAPPE_SITE_URL", "http://frontend:8080")
        self.api_key = os.getenv("FRAPPE_API_KEY")
        self.api_secret = os.getenv("FRAPPE_API_SECRET")
        
        self.headers = {
            "Authorization": f"token {self.api_key}:{self.api_secret}",
            "Content-Type": "application/json"
        }
    
    def update_lead(self, email: str, data: Dict) -> bool:
        """Create or update lead in Frappe CRM"""
        
        # Check if lead exists
        try:
            existing_lead = self._get_lead(email)
            
            if existing_lead:
                # Update existing
                response = requests.put(
                    f"{self.site_url}/api/resource/Lead/{existing_lead['name']}",
                    headers=self.headers,
                    json=data
                )
            else:
                # Create new
                response = requests.post(
                    f"{self.site_url}/api/resource/Lead",
                    headers=self.headers,
                    json=data
                )
            
            return response.status_code in [200, 201]
            
        except Exception as e:
            print(f"Frappe update error: {e}")
            return False
    
    def _get_lead(self, email: str) -> Optional[Dict]:
        """Get existing lead by email"""
        try:
            response = requests.get(
                f"{self.site_url}/api/resource/Lead",
                headers=self.headers,
                params={"filters": [["email_id", "=", email]]}
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["data"][0] if data["data"] else None
                
        except Exception as e:
            print(f"Lead fetch error: {e}")
            return None
    
    def log_conversation(self, lead_email: str, conversation_data: Dict):
        """Log conversation as a Note linked to Lead"""
        try:
            requests.post(
                f"{self.site_url}/api/resource/Note",
                headers=self.headers,
                json={
                    "title": f"AI Conversation - {lead_email}",
                    "content": str(conversation_data),
                    "custom_linked_lead": lead_email
                }
            )
        except Exception as e:
            print(f"Conversation log error: {e}")

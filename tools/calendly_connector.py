import os
from typing import Optional


class CalendlyConnector:
    def __init__(self):
        self.api_key = os.getenv("CALENDLY_API_KEY")
        self.event_type_url = os.getenv("CALENDLY_EVENT_TYPE_URL")
    
    def generate_link(self, email: Optional[str] = None, name: Optional[str] = None) -> str:
        """Generate Calendly scheduling link with pre-filled data"""
        
        base_url = self.event_type_url or "https://calendly.com/your-team/demo"
        
        params = []
        if email:
            params.append(f"email={email}")
        if name:
            params.append(f"name={name}")
        
        if params:
            return f"{base_url}?{'&'.join(params)}"
        return base_url

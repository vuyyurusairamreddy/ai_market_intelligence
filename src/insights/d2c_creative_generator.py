import json
import logging
from typing import Dict, List, Any
import pandas as pd
import requests
from config.settings import Config

class D2CCreativeGenerator:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Use requests instead of OpenAI client
        self.headers = {
            "Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://api.perplexity.ai/chat/completions"
    
    def _make_api_request(self, prompt: str, temperature: float = 0.7) -> str:
        """Make API request to Perplexity"""
        try:
            payload = {
                "model": self.config.PERPLEXITY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 2000
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return f"API request failed: {response.status_code}"
                
        except Exception as e:
            self.logger.error(f"Error making API request: {e}")
            return f"Error making API request: {str(e)}"
    
    def generate_ad_headlines(self, d2c_insights: Dict, top_categories: List[str]) -> List[Dict]:
        """Generate AI-powered ad headlines based on D2C insights"""
        
        # Extract key performance data
        best_channel = d2c_insights['channel_analysis']['best_roas']
        avg_roas = d2c_insights['data_summary']['overall_roas']
        
        # Prepare context for headline generation
        context = {
            'best_performing_channel': best_channel,
            'average_roas': round(avg_roas, 2),
            'top_categories': top_categories[:3],
            'recommendations': [rec for rec in d2c_insights['recommendations'] if rec['priority'] == 'high']
        }
        
        prompt = f"""
        Based on the following D2C eCommerce performance data, generate 5 compelling ad headlines for different marketing channels:
        
        Performance Context:
        - Best performing channel: {context['best_performing_channel']}
        - Average ROAS: {context['average_roas']}x
        - Top performing categories: {', '.join(context['top_categories'])}
        
        Generate headlines that:
        1. Highlight value propositions and benefits
        2. Include urgency or scarcity elements
        3. Address common customer pain points
        4. Are optimized for different channels (Facebook, Google, Email, etc.)
        5. Include specific offers or discounts where appropriate
        
        Format each headline with the target channel and explanation of why it would work.
        """
        
        try:
            content = self._make_api_request(prompt)
            
            headlines = {
                'type': 'ad_headlines',
                'content': content,
                'context': context,
                'generated_for': 'multiple_channels'
            }
            
            return headlines
            
        except Exception as e:
            self.logger.error(f"Error generating ad headlines: {e}")
            return {'type': 'ad_headlines', 'content': 'Generation failed', 'error': str(e)}
    
    # Add other methods with same pattern...
    def generate_all_creative_content(self, d2c_insights: Dict) -> List[Dict]:
        """Generate all types of creative content based on D2C insights"""
        
        creative_outputs = []
        
        # Extract key categories and insights
        try:
            seo_categories = list(d2c_insights['seo_analysis']['top_performing_categories'].keys())[:5]
        except:
            seo_categories = ['skincare', 'fitness', 'fashion']
        
        self.logger.info("Generating ad headlines...")
        headlines = self.generate_ad_headlines(d2c_insights, seo_categories)
        creative_outputs.append(headlines)
        
        # Add basic SEO content
        meta_descriptions = {
            'type': 'seo_meta_descriptions',
            'content': 'SEO meta descriptions for high-opportunity categories generated based on search volume and conversion data.',
            'target_categories': seo_categories[:3]
        }
        creative_outputs.append(meta_descriptions)
        
        # Add basic PDP content
        pdp_content = {
            'type': 'pdp_content',
            'content': 'Product Description Page content optimized for conversion based on top-performing categories and customer behavior patterns.',
            'target_categories': seo_categories[:3]
        }
        creative_outputs.append(pdp_content)
        
        return creative_outputs

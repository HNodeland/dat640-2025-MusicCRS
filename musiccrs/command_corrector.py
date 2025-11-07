"""
OLLAMA-based Command Corrector for Unknown Intents (R7.3)

This module uses OLLAMA to suggest corrections for unrecognized commands,
ensuring the system only suggests valid intents that exist in the chatbot.

Features:
- Detect unknown intents
- Use OLLAMA to suggest corrections to known commands
- Constrained output (only suggest valid intents)
- Fallback to fuzzy matching if OLLAMA unavailable

Usage:
    corrector = CommandCorrector()
    suggestion = corrector.suggest_command("addd shape of you")
    # Returns: {"corrected": "add", "full_command": "add shape of you", "confidence": 0.9}
"""

import os
import time
from typing import Dict, Optional, List
from dotenv import load_dotenv

try:
    import ollama
except ImportError:
    ollama = None

load_dotenv()


class CommandCorrector:
    """Corrects unknown commands using OLLAMA."""
    
    # Valid commands in the system
    VALID_COMMANDS = [
        'add', 'remove', 'show', 'view', 'play', 'recommend', 
        'create', 'switch', 'clear', 'ask', 'help'
    ]
    
    # Command descriptions for OLLAMA context
    COMMAND_DESCRIPTIONS = {
        'add': 'Add a track to the playlist',
        'remove': 'Remove a track from the playlist',
        'show': 'Show/view the current playlist',
        'view': 'View the current playlist',
        'play': 'Play a track from the playlist',
        'recommend': 'Get song recommendations',
        'create': 'Create a new playlist',
        'switch': 'Switch to a different playlist',
        'clear': 'Clear the current playlist',
        'ask': 'Ask a question about music',
        'help': 'Show available commands'
    }
    
    def __init__(self):
        """Initialize command corrector."""
        self.ollama_host = os.getenv("OLLAMA_HOST")
        self.ollama_model = os.getenv("OLLAMA_MODEL")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY")
        
        self.ollama_available = False
        if ollama and self.ollama_host and self.ollama_model:
            try:
                headers = {}
                if self.ollama_api_key:
                    headers["Authorization"] = f"Bearer {self.ollama_api_key}"
                # CRITICAL: Content-Type header is required by the UiS OLLAMA server
                headers["Content-Type"] = "application/json"
                
                self.client = ollama.Client(
                    host=self.ollama_host,
                    headers=headers
                )
                self.ollama_available = True
                print("Command corrector: OLLAMA available")
            except Exception as e:
                print(f"Command corrector: OLLAMA unavailable ({e})")
                self.ollama_available = False
        else:
            print("Command corrector: OLLAMA not configured")
    
    def suggest_command(self, user_input: str) -> Optional[Dict]:
        """
        Suggest a corrected command for unknown input.
        
        Args:
            user_input: The unknown/misspelled command
            
        Returns:
            Dictionary with:
            - corrected: The corrected command word
            - full_command: Full corrected command with entity
            - confidence: Confidence score (0-1)
            - suggestion: Human-readable suggestion
            - latency_ms: Time taken to generate suggestion
            - method: 'ollama' or 'fuzzy'
            
            Returns None if no suggestion available
        """
        if not user_input:
            return None
        
        start_time = time.time()
        
        # Try OLLAMA first
        if self.ollama_available:
            result = self._suggest_with_ollama(user_input, start_time)
            if result:
                return result
        
        # Fallback to simple fuzzy matching
        return self._suggest_with_fuzzy(user_input, start_time)
    
    def _suggest_with_ollama(self, user_input: str, start_time: float) -> Optional[Dict]:
        """
        Use OLLAMA to suggest command correction.
        
        Strategy:
        - Provide OLLAMA with the list of valid commands
        - Ask it to identify the most likely intended command
        - Constrain output to only valid commands
        """
        # Build prompt
        commands_list = "\n".join([
            f"- {cmd}: {self.COMMAND_DESCRIPTIONS[cmd]}"
            for cmd in self.VALID_COMMANDS
        ])
        
        prompt = f"""You are a command corrector for a music chatbot. The user typed a command that wasn't recognized.

Valid commands:
{commands_list}

User typed: "{user_input}"

Task: Identify which valid command the user most likely intended. Consider:
1. Spelling similarity
2. Command purpose/meaning
3. Context from any additional text

Response format (JSON):
{{
  "command": "<one of the valid commands above>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}

If no valid command seems likely, respond with:
{{
  "command": null,
  "confidence": 0.0,
  "reasoning": "Cannot match to any valid command"
}}

Respond ONLY with valid JSON, no other text."""
        
        try:
            ollama_start = time.time()
            response = self.client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}  # Lower temperature for more consistent output
            )
            ollama_latency = (time.time() - ollama_start) * 1000
            
            # Parse response
            import json
            response_text = response['message']['content'].strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result = json.loads(response_text)
            
            command = result.get('command')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            
            total_latency = (time.time() - start_time) * 1000
            
            if command and command in self.VALID_COMMANDS and confidence > 0.3:
                # Extract entity (text after first word)
                tokens = user_input.split(maxsplit=1)
                entity = tokens[1] if len(tokens) > 1 else ""
                
                full_command = f"{command} {entity}".strip()
                
                return {
                    'corrected': command,
                    'full_command': full_command,
                    'confidence': confidence,
                    'suggestion': f"Did you mean: {command}? ({reasoning})",
                    'latency_ms': total_latency,
                    'ollama_latency_ms': ollama_latency,
                    'method': 'ollama'
                }
            
            return None
            
        except Exception as e:
            print(f"OLLAMA command correction error: {e}")
            return None
    
    def _suggest_with_fuzzy(self, user_input: str, start_time: float) -> Optional[Dict]:
        """
        Fallback fuzzy matching for command correction.
        Uses simple Levenshtein-like matching.
        """
        import difflib
        
        # Extract first word as potential command
        tokens = user_input.lower().split(maxsplit=1)
        if not tokens:
            return None
        
        first_word = tokens[0]
        entity = tokens[1] if len(tokens) > 1 else ""
        
        # Find closest match
        matches = difflib.get_close_matches(
            first_word,
            self.VALID_COMMANDS,
            n=1,
            cutoff=0.6
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if matches:
            corrected = matches[0]
            full_command = f"{corrected} {entity}".strip()
            
            # Calculate simple similarity
            similarity = difflib.SequenceMatcher(None, first_word, corrected).ratio()
            
            return {
                'corrected': corrected,
                'full_command': full_command,
                'confidence': similarity,
                'suggestion': f"Did you mean: {corrected}?",
                'latency_ms': latency_ms,
                'method': 'fuzzy'
            }
        
        return None
    
    def format_suggestion_html(self, suggestion: Dict) -> str:
        """
        Format suggestion as HTML for display.
        
        Args:
            suggestion: Suggestion dictionary from suggest_command()
            
        Returns:
            HTML string
        """
        if not suggestion:
            return ""
        
        confidence = suggestion['confidence']
        confidence_pct = int(confidence * 100)
        latency_ms = suggestion.get('latency_ms', 0)
        method = suggestion.get('method', 'unknown')
        ollama_latency = suggestion.get('ollama_latency_ms', 0)
        
        # Build timing info string
        if method == 'ollama':
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[OLLAMA: {ollama_latency:.0f}ms, total: {latency_ms:.0f}ms]</span>"
        else:
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[{method}: {latency_ms:.0f}ms]</span>"
        
        html = f"""
        <div style="padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; margin: 10px 0;">
            <strong>❓ Unknown command</strong><br/>
            {suggestion['suggestion']}<br/>
            <code>{suggestion['full_command']}</code>
            <span style="opacity: 0.7; font-size: 0.9em;">(confidence: {confidence_pct}%)</span>{timing_info}
        </div>
        """
        
        return html


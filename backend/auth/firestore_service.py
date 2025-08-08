"""Firebase Firestore service for robot port configurations"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
try:
    from firebase_admin import firestore  # type: ignore
    FIREBASE_LIB_AVAILABLE = True
except Exception:
    firestore = None  # type: ignore
    FIREBASE_LIB_AVAILABLE = False

from .firebase_auth import firebase_initialized

class FirestoreService:
    """Service for managing robot port configurations in Firebase Firestore"""
    
    def __init__(self):
        if not (FIREBASE_LIB_AVAILABLE and firebase_initialized):
            raise Exception("Firebase not available (optional mode). Configure Firebase to enable Firestore features.")
        
        self.db = firestore.client()
        self.collection_name = 'robot_port_configurations'
    
    def save_robot_configuration(self, user_email: str, robot_type: str, 
                                leader_port: Optional[str] = None, 
                                follower_port: Optional[str] = None) -> Dict[str, Any]:
        """
        Save or update robot port configuration for a user
        
        Args:
            user_email: User's email address
            robot_type: Type of robot (Koch, SO-100, etc.)
            leader_port: Leader arm USB port
            follower_port: Follower arm USB port
            
        Returns:
            Dict containing the saved configuration
        """
        if not leader_port and not follower_port:
            raise ValueError("At least one port (leader or follower) must be provided")
        
        # Create document ID based on user and robot type
        doc_id = f"{user_email}_{robot_type}".replace('@', '_').replace('.', '_')
        
        # Prepare configuration data
        config_data = {
            'user_email': user_email,
            'robot_type': robot_type,
            'leader_port': leader_port,
            'follower_port': follower_port,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'is_active': True
        }
        
        # Check if document exists
        doc_ref = self.db.collection(self.collection_name).document(doc_id)
        doc = doc_ref.get()
        
        if doc.exists:
            # Update existing document
            doc_ref.update(config_data)
        else:
            # Create new document
            config_data['created_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(config_data)
        
        # Return the saved data (with timestamps converted)
        saved_doc = doc_ref.get()
        result = saved_doc.to_dict()
        result['id'] = doc_id
        
        # Convert timestamps to strings for JSON serialization
        if result.get('created_at'):
            result['created_at'] = result['created_at'].isoformat()
        if result.get('updated_at'):
            result['updated_at'] = result['updated_at'].isoformat()
        
        return result
    
    def get_user_configurations(self, user_email: str) -> List[Dict[str, Any]]:
        """
        Get all active robot configurations for a user
        
        Args:
            user_email: User's email address
            
        Returns:
            List of configuration dictionaries
        """
        try:
            # Query configurations for the user
            configs_ref = self.db.collection(self.collection_name)
            query = configs_ref.where('user_email', '==', user_email).where('is_active', '==', True)
            
            configurations = []
            for doc in query.stream():
                config = doc.to_dict()
                config['id'] = doc.id
                
                # Convert timestamps to strings for JSON serialization
                if config.get('created_at'):
                    config['created_at'] = config['created_at'].isoformat()
                if config.get('updated_at'):
                    config['updated_at'] = config['updated_at'].isoformat()
                
                configurations.append(config)
            
            return configurations
            
        except Exception as e:
            print(f"Error getting user configurations: {e}")
            return []
    
    def get_configuration(self, user_email: str, robot_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific robot configuration for a user
        
        Args:
            user_email: User's email address
            robot_type: Type of robot
            
        Returns:
            Configuration dictionary or None if not found
        """
        try:
            doc_id = f"{user_email}_{robot_type}".replace('@', '_').replace('.', '_')
            doc_ref = self.db.collection(self.collection_name).document(doc_id)
            doc = doc_ref.get()
            
            if doc.exists and doc.to_dict().get('is_active', True):
                config = doc.to_dict()
                config['id'] = doc.id
                
                # Convert timestamps to strings
                if config.get('created_at'):
                    config['created_at'] = config['created_at'].isoformat()
                if config.get('updated_at'):
                    config['updated_at'] = config['updated_at'].isoformat()
                
                return config
            
            return None
            
        except Exception as e:
            print(f"Error getting configuration: {e}")
            return None
    
    def delete_configuration(self, user_email: str, robot_type: str) -> bool:
        """
        Soft delete a robot configuration (mark as inactive)
        
        Args:
            user_email: User's email address
            robot_type: Type of robot
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = f"{user_email}_{robot_type}".replace('@', '_').replace('.', '_')
            doc_ref = self.db.collection(self.collection_name).document(doc_id)
            
            doc_ref.update({
                'is_active': False,
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            
            return True
            
        except Exception as e:
            print(f"Error deleting configuration: {e}")
            return False
    
    def get_all_configurations(self) -> List[Dict[str, Any]]:
        """
        Get all active robot configurations (admin function)
        
        Returns:
            List of all configuration dictionaries
        """
        try:
            configs_ref = self.db.collection(self.collection_name)
            query = configs_ref.where('is_active', '==', True)
            
            configurations = []
            for doc in query.stream():
                config = doc.to_dict()
                config['id'] = doc.id
                
                # Convert timestamps to strings
                if config.get('created_at'):
                    config['created_at'] = config['created_at'].isoformat()
                if config.get('updated_at'):
                    config['updated_at'] = config['updated_at'].isoformat()
                
                configurations.append(config)
            
            return configurations
            
        except Exception as e:
            print(f"Error getting all configurations: {e}")
            return []

# Global instance
firestore_service = None

def get_firestore_service() -> FirestoreService:
    """Get global Firestore service instance"""
    global firestore_service
    if firestore_service is None:
        firestore_service = FirestoreService()
    return firestore_service
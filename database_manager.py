import sqlite3
import hashlib
import streamlit as st
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    is_approved BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check if admin exists, if not create default admin
            cursor.execute("SELECT * FROM users WHERE role = 'admin'")
            if not cursor.fetchone():
                # Create default admin (change this password immediately after first login)
                admin_password = hashlib.sha256("Admin@123".encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO users (email, password, full_name, role, is_approved)
                    VALUES (?, ?, ?, ?, ?)
                ''', ("admin@hospital.ac.ke", admin_password, "System Admin", "admin", 1))
            
            conn.commit()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, email, password, full_name):
        """Register a new user"""
        try:
            # Validate email domain
            if not email.endswith('@hospital.ac.ke'):
                return False, "Email must end with @hospital.ac.ke"
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                hashed_password = self.hash_password(password)
                cursor.execute('''
                    INSERT INTO users (email, password, full_name, role, is_approved)
                    VALUES (?, ?, ?, 'user', 0)
                ''', (email, hashed_password, full_name))
                conn.commit()
                return True, "Registration successful! Waiting for admin approval."
        except sqlite3.IntegrityError:
            return False, "Email already exists!"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def login_user(self, email, password):
        """Authenticate user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            hashed_password = self.hash_password(password)
            cursor.execute('''
                SELECT id, email, full_name, role, is_approved 
                FROM users 
                WHERE email = ? AND password = ?
            ''', (email, hashed_password))
            user = cursor.fetchone()
            
            if user:
                if not user[4] and user[3] != 'admin':  # is_approved = 0 and not admin
                    return None, "Your account is pending admin approval."
                return user, "Login successful"
            return None, "Invalid email or password"
    
    def get_pending_users(self):
        """Get all users pending approval"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, email, full_name, created_at 
                FROM users 
                WHERE is_approved = 0 AND role = 'user'
                ORDER BY created_at
            ''')
            return cursor.fetchall()
    
    def get_all_users(self):
        """Get all users"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, email, full_name, role, is_approved, created_at 
                FROM users 
                ORDER BY created_at
            ''')
            return cursor.fetchall()
    
    def approve_user(self, user_id):
        """Approve a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET is_approved = 1 
                WHERE id = ? AND role = 'user'
            ''', (user_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_user(self, user_id):
        """Delete a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Prevent deleting the last admin
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
            user_role = cursor.fetchone()
            
            if user_role and user_role[0] == 'admin' and admin_count <= 1:
                return False, "Cannot delete the last admin account"
            
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0, "User deleted successfully"
    
    def change_user_role(self, user_id, new_role):
        """Change user role (admin only)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET role = ? 
                WHERE id = ?
            ''', (new_role, user_id))
            conn.commit()
            return cursor.rowcount > 0

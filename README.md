# 🏥 ICD-11 Medical Notes Classifier

An AI-powered system for classifying medical notes into ICD-11 chapters using fine-tuned BioBERT, with user authentication and admin management.

## ✨ Features

- **ICD-11 Classification**: Classifies medical notes into 22 ICD-11 chapters using BioBERT
- **User Authentication**: Secure login system with email domain restriction (@hospital.ac.ke)
- **Admin Dashboard**: Approve/delete users, manage roles, view statistics
- **Batch Processing**: Process multiple notes via CSV upload
- **High Accuracy**: 100% test accuracy on 11,000 training samples

## 🚀 Live Demo

[Your Streamlit App URL]

## 🔐 Default Admin Credentials

**⚠️ IMPORTANT: Change these immediately after first login!**

- **Email:** `admin@hospital.ac.ke`
- **Password:** `Admin@123`

## 📋 User Registration

New users must register with an email ending in `@hospital.ac.ke`. Accounts require admin approval before accessing the classifier.

## 🛠️ Local Development

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/icd11-classifier.git
cd icd11-classifier

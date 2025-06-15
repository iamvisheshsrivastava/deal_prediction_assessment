# Deal Prediction Assessment

## Problem Statement  
You are required to build an end-to-end workflow that predicts whether a deal will be won or lost based on sales activities data. The system must analyse patterns in customer interactions and activities to make accurate binary predictions about deal outcomes.

## Given Data  
• won_deals.json - Successfully closed deals  
• lost_deals.json - Failed deals  
• activities.json - All activities associated with deals

### Activity Types  
Each deal has 5 types of activities:  
1. Email - Email communications  
2. Meetings - Scheduled meetings  
3. Tasks - Action items and follow-ups  
4. Notes - Free-form observations  
5. Calls - Phone conversations

## Expected System Capabilities  
The workflow must:  
• Process and integrate the three JSON datasets  
• Extract meaningful features from activities data  
• Train a binary classification model to predict won/lost outcomes  
• Provide end-to-end prediction capability with innovation in approach

## Technical Requirements

### Data Processing  
• Load and validate JSON files (won_deals.json, lost_deals.json, activities.json)  
• Merge deals with their corresponding activities  
• Handle data cleaning and preprocessing  

### Feature Engineering  
• Quantitative Features: Activity counts, frequency patterns, timing analysis  
• Temporal Features: Activity progression over deal lifecycle  

### Machine Learning Model  
• Implement binary classification for won/lost prediction  
• Use activities-based features as input  
• Ensure model interpretability for business insights  

## Innovation Requirements  
• Creative feature engineering using both structured and unstructured data  
• End-to-end automated workflow  
• Unique insights extraction from sales activities

## Core Deliverables  
GitHub Link with Exploratory Data Analysis (EDA) and testing notebooks, Python scripts for data preprocessing and model architecture, the trained model with instructions for loading and inference, a comprehensive README file, and detailed documentation outlining observations, methodologies, performance evaluation, and suggestions for future enhancements.

### 1. Data Processing Pipeline  
• JSON data loader and validator  
• Data integration and cleaning modules  
• Deal-activity mapping system  

### 2. Feature Engineering System  
• Activity-based feature extraction  
• Feature selection and optimization  

### 3. Binary Classification Model  
• Model training pipeline  
• Won/lost prediction system  
• Model evaluation and validation  

### 4. End-to-End Workflow  
• Complete prediction pipeline  
• Model performance documentation

## Submission  
You should submit a GitHub link of your deliverables.

---

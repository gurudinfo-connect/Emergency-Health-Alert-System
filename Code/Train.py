import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

print("=== STARTING FINAL TRAINING PROCESS ===")

# ==========================================
# 1. HEART DISEASE MODULE
# ==========================================
print("\n--- Processing Heart Disease Model ---")
try:
    df_heart = pd.read_csv('heart.csv')
    df_heart.columns = df_heart.columns.str.strip()

    if 'num' in df_heart.columns:
        df_heart['target'] = df_heart['num'].apply(lambda x: 1 if x > 0 else 0)
        df_heart.drop('num', axis=1, inplace=True)
    elif 'target' not in df_heart.columns:
        pass

    for col in ['id', 'dataset']:
        if col in df_heart.columns:
            df_heart.drop(col, axis=1, inplace=True)

    for col in df_heart.columns:
        if df_heart[col].dtype == 'object':
            le = LabelEncoder()
            df_heart[col] = le.fit_transform(df_heart[col].astype(str))

    imputer = SimpleImputer(strategy='mean')
    if 'target' in df_heart.columns:
        X_heart = df_heart.drop(['target'], axis=1)
        y_heart = df_heart['target']
        
        X_heart = pd.DataFrame(imputer.fit_transform(X_heart), columns=X_heart.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
        
        rf_heart = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_heart.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, y_pred=rf_heart.predict(X_test))
        print(f"✅ Heart Model Accuracy: {acc * 100:.2f}%")
        
        with open('models/heart_model.pkl', 'wb') as f:
            pickle.dump(rf_heart, f)

except Exception as e:
    print(f"❌ Error training Heart Model: {e}")


# ==========================================
# 2. KIDNEY DISEASE MODULE
# ==========================================
print("\n--- Processing Kidney Disease Model ---")
try:
    df_kidney = pd.read_csv('kidney_disease.csv')
    
    if 'classification' in df_kidney.columns:
        df_kidney['classification'] = df_kidney['classification'].astype(str).map(
            {'ckd': 1, 'ckd\t': 1, 'notckd': 0, 'notckd\t': 0}
        )
        
        if 'id' in df_kidney.columns:
            df_kidney.drop('id', axis=1, inplace=True)
            
        cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for col in cat_cols:
            if col in df_kidney.columns:
                df_kidney[col] = df_kidney[col].astype(str).str.strip()
                le = LabelEncoder()
                df_kidney[col] = le.fit_transform(df_kidney[col])

        for col in df_kidney.columns:
            df_kidney[col] = pd.to_numeric(df_kidney[col], errors='coerce')
            
        df_kidney.fillna(df_kidney.mean(), inplace=True)
        
        X_kidney = df_kidney.drop('classification', axis=1)
        y_kidney = df_kidney['classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)
        
        rf_kidney = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_kidney.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, y_pred=rf_kidney.predict(X_test))
        print(f"✅ Kidney Model Accuracy: {acc * 100:.2f}%")
        
        with open('models/kidney_model.pkl', 'wb') as f:
            pickle.dump(rf_kidney, f)

except Exception as e:
    print(f"❌ Error training Kidney Model: {e}")


# ==========================================
# 3. SKIN CANCER MODULE (Final Optimized Strategy)
# ==========================================
print("\n--- Processing Skin Cancer Model ---")

TRAIN_DIR = 'Split_smol/train'
VAL_DIR = 'Split_smol/val'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

if os.path.exists(TRAIN_DIR):
    # Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    
    val_gen = None
    if os.path.exists(VAL_DIR):
        val_gen = val_datagen.flow_from_directory(
            VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
        )

    # Calculate Class Weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    train_class_weights = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {train_class_weights}")

    # Build Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False 

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5), 
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early Stopping: Stop if not improving for 8 epochs, and restore the BEST weights
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

    print("Training Model (Max 30 Epochs)...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=30,  # Longer training time
        class_weight=train_class_weights,
        callbacks=[early_stop] if val_gen else None,
        verbose=1
    )
    
    final_acc = history.history['val_accuracy'][-1] if val_gen else history.history['accuracy'][-1]
    print(f"✅ Skin Model Final Accuracy: {final_acc * 100:.2f}%")
    
    model.save('models/skin_model.h5')
    print("Detected Classes:", train_gen.class_indices)

else:
    print("❌ Error: 'Split_smol/train' directory not found.")

print("\n=== ALL TASKS COMPLETED ===")

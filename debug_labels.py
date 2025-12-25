from src.preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.load_nsl_kdd_dataset('data/KDDTrain+.csv', 'data/KDDTest+.csv')
print("Label column unique values:")
print(train_df['label'].unique())
print("\nLabel value counts:")
print(train_df['label'].value_counts())

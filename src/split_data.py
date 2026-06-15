from sklearn.model_selection import train_test_split

def split_data(data, random_state=42):
    train, temp = train_test_split(data, test_size=0.2, random_state = random_state, stratify=data["label"])
    validation, test = train_test_split(temp, test_size=0.5, random_state=random_state, stratify=temp["label"])
    return train,validation,test

from utils import create_data, add_constant, train_model

X, y = create_data()

X = add_constant(X)

model = train_model(X, y)

print(model.summary())

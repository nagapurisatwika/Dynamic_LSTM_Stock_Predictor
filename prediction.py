import numpy as np

def predict_future(model, last_sequence, scaler, days):

    predictions = []

    current_seq = last_sequence.copy()

    for _ in range(days):

        pred = model.predict(current_seq.reshape(1,len(current_seq),1), verbose=0)

        predictions.append(pred[0][0])

        current_seq = np.append(current_seq[1:], pred)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    )

    return predictions
from deepface import DeepFace
from time import perf_counter

def get_emotions(user_image_path : str | None) -> dict:
    """
    Функция распознает эмоции человека с фотографии.
    На вход принимает путь к изображению в формате str.

    Возвращает словарь вида:
    {'emotions': {'angry': np.float32(0),
                  'disgust': np.float32(0),
                  'fear': np.float32(0),
                  'happy': np.float32(0),
                  'sad': np.float32(0),
                  'neutral': np.float32(0)},
     'dominant_emotion': 'angry',
     'inference_time': 0.27134099999966566} - в случае успешного распознавания лица.
    
    В противном случае:
    {'emotions': {},
     'dominant_emotion': '',
     'inference_time': 0}
    """

    result = {'emotions': {},
              'dominant_emotion': '',
              'inference_time': 0.0}

    if not isinstance(user_image_path, str):
        return result

    start_time = perf_counter()

    try:
        detected_emotions = DeepFace.analyze(img_path = user_image_path,
                              actions = ["emotion"])[0]
    except Exception:
        return result

    end_time = perf_counter()

    del detected_emotions['emotion']['surprise']

    result['emotions'] = detected_emotions['emotion']
    result['dominant_emotion'] = detected_emotions['dominant_emotion']
    result['inference_time'] = end_time - start_time

    return result


# Тестовая проверка
if __name__ == "__main__":
    print(get_emotions('.test_images/test.png'))
    print(get_emotions('.test_images/test2.png'))

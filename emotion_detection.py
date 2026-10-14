from deepface import DeepFace

def get_emotions(user_image_path : str | None) -> dict:
    """Возвращает словарь эмоций с вероятностями с фотографии"""

    if user_image_path is None:
        return {}

    result = DeepFace.analyze(img_path = user_image_path,
                              actions = ["emotion"])

    return result[0]['emotion']

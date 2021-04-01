from functions import *
import pandas as pd
import joblib

""" LOAD MODELS"""

load_model_all = joblib.load(open("../2.Models/model_all.sav", "rb"))
load_model_val = joblib.load(open("../2.Models/model_fan.sav", "rb"))
load_model_pump = joblib.load(open("../2.Models/model_pump.sav", "rb"))
load_model_slider = joblib.load(open("../2.Models/model_slider.sav", "rb"))
load_model_valve = joblib.load(open("../2.Models/model_valve.sav", "rb"))


""" IMPORT SOUND TEST """
sound = load_sound("../Exploration_phase/sounds_sample/normal_fan.wav")

""" GET FEATURES"""

features = (pd.DataFrame(get_features(sound))).T

""" PREDICT"""

prediction = load_model_all.predict(features)

print(prediction)

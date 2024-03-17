# result["text"]
import pandas as pd
import numpy as np
from pytube import YouTube

# dataset = pd.read_csv("/content/drive/MyDrive/Datasets TCC Etapa 1/Etapa 3 - Misto/Pesquisa/ Dataset_Misto_Pesquisa_Extraido_Total.csv")
# # dataset["Texto"] = ''

# for pos, data in dataset.iterrows():
#   yt = YouTube(data["Link"])
#   if pd.isna(dataset["Texto"][pos]) == True:
#     print(pos)
#     try:
#       yt.streams.filter(file_extension='mp4')
#     except VideoUnavailable:
#       dataset["Texto"][pos] = np.nan
#       pass
#     else:
#       stream = yt.streams.get_by_itag(139)
#       stream.download('', "GoogleImagen.mp4")

#       model = whisper.load_model("base")
#       result = model.transcribe("GoogleImagen.mp4", fp16=False)
#       os.remove("GoogleImagen.mp4")
#       i = 0
#       texto = ''
#       for time in result["segments"]:
#         if time["end"] <= 1200:
#           texto = texto + time["text"]
#         elif time["end"] >= 1200:
#           break
#         i = i + 1
#       dataset["Texto"][pos] = texto
#       dataset.to_csv("/content/drive/MyDrive/Datasets TCC Etapa 1/Etapa 3 - Misto/Pesquisa/ Dataset_Misto_Pesquisa_Extraido_Total.csv", index=False)



dataset = pd.read_csv("/content/drive/MyDrive/Datasets TCC Etapa 1/Etapa 3 - Desinformativo/Pesquisa/Dataset_Desinformativo_Pesquisa_Extraido_Total.csv")
# dataset["Texto"] = ''
for pos, data in dataset.iterrows():
  yt = YouTube(data["Link"], use_oauth=True, allow_oauth_cache=True)
  if pd.isna(dataset["Texto"][pos]) == True:
    print(pos)
    try:
      try:
        yt.streams.filter(file_extension='mp4')
        key_error = 0
      except KeyError:
        print("Key Error")
        key_error = 1
        dataset["Texto"][pos] = np.nan
        pass
    except VideoUnavailable:
      dataset["Texto"][pos] = np.nan
      key_error = 0
      pass
    else:
      if key_error == 1:
        key_error = 0
        pass
      else:
        if yt.length <=1200:
          stream = yt.streams.get_by_itag(139)
          stream.download('', "GoogleImagen.mp4")

          model = whisper.load_model("base")
          result = model.transcribe("GoogleImagen.mp4", fp16=False)
          os.remove("GoogleImagen.mp4")
          i = 0
          texto = ''
          for time in result["segments"]:
            if time["end"] <= 1200:
              texto = texto + time["text"]
            elif time["end"] >= 1200:
              break
            i = i + 1
          dataset["Texto"][pos] = texto
          dataset.to_csv("/content/drive/MyDrive/Datasets TCC Etapa 1/Etapa 3 - Desinformativo/Pesquisa/Dataset_Desinformativo_Pesquisa_Extraido_Total.csv", index=False)


# print(dataset["Link"][0])
# len(dataset.index)
# dataset["Texto"] = ''
# dataset = dataset.drop(["Texto"], axis="columns")

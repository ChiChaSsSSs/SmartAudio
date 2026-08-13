import librosa
import librosa.display
import matplotlib.pyplot as plt # Biblioteka za crtanje grafika
import numpy as np # Biblioteka za rad sa nizovima i matricama

FRAME_SIZE = 1024 # Velicina frejma za analizu audio signala

# Putanje do audio fajlova
debussy_path = "./AudioFiles/debussy.wav"
duke_path = "./AudioFiles/duke.wav"
redhot_path = "./AudioFiles/redhot.wav"

# Ucitavanje audio fajlova
debussy, sr = librosa.load(debussy_path) # Metoda za ucitanje audio fajla (ucitava se signal i sampling rate)
duke, _ = librosa.load(duke_path)
redhot, _ = librosa.load(redhot_path)

# Analiza audio signala Debussy
number_of_samples_debussy = debussy.size # Broj uzoraka u audio signalu
print(f"Broj uzoraka je {number_of_samples_debussy}") # Ispisuje broj uzoraka u audio signalu
sample_duration = 1 / sr # Vreme trajanja jednog uzorka
print(f"Vreme trajanja jednog uzorka je {sample_duration:.6f} s") # Ispisuje vreme trajanja jednog uzorka
duration_debussy = number_of_samples_debussy * sample_duration # Ukupno trajanje audio signala
print(f"Ukupno trajanje audio signala je {duration_debussy:.2f} s") # Ispisuje ukupno trajanje audio signala

# Vizuelizacija audio signala Debussy
plt.figure(figsize=(15, 17)) # Kreira graficki prozor dimenzija 15x17 inča
plt.subplot(3, 1, 1) # Kreira subplot (3 reda, 1 kolona, prvi subplot)
librosa.display.waveshow(debussy, alpha=0.5) # Crta waveform audio signala
plt.title("Debussy") # Postavlja naslov grafa
plt.ylim(-1, 1) # Postavlja granice y-ose
# Vizuelizacija audio signala Duke
plt.subplot(3, 1, 2)
librosa.display.waveshow(duke, alpha=0.5)
plt.title("Duke")
plt.ylim(-1, 1)
# Vizuelizacija audio signala Red Hot Chili Peppers
plt.subplot(3, 1, 3)
librosa.display.waveshow(redhot, alpha=0.5)
plt.title("Red Hot Chili Peppers")
plt.ylim(-1, 1)
plt.subplots_adjust(hspace=0.5) # Podesava razmak izmedju subplota
plt.show() # Prikazuje grafike

# Funkcija za izracunavanje amplitude envelope audio signala
def amplitude_envelope(signal, frame_size):
    amplitude_envelope = []
    for i in range(0, len(signal), frame_size // 2): # Skok je polovina velicine frejma zbog preklapanja frejmova
        current_frame_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)
    return np.array(amplitude_envelope)

# Funkcija za izracunavanje amplitude envelope audio signala sa fancy metodom
def fancy_amplitude_envelope(signal, frame_size):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), frame_size // 2)])

# Izracunavanje amplitude envelope za svaki audio signal
ae_debussy = amplitude_envelope(debussy, FRAME_SIZE)
print(f"Broj frejmova za Debussy je {ae_debussy.size}")
ae_duke = amplitude_envelope(duke, FRAME_SIZE)
print(f"Broj frejmova za Duke je {ae_duke.size}")
ae_redhot = amplitude_envelope(redhot, FRAME_SIZE)
print(f"Broj frejmova za Red Hot Chili Peppers je {ae_redhot.size}")

# Vizuelizacija amplitude envelope za svaki audio signal
frames = range(0, len(ae_debussy))
t = librosa.frames_to_time(frames, sr=sr, hop_length=FRAME_SIZE // 2) # Konvertuje frejmove u vreme
plt.figure(figsize=(15, 17))
plt.subplot(3, 1, 1)
#librosa.display.waveshow(debussy, alpha=0.5)
plt.plot(t, ae_debussy, color='r', alpha=0.8)
plt.title("Debussy")
plt.ylim(-1, 1)
plt.subplot(3, 1, 2)
#librosa.display.waveshow(duke, alpha=0.5)
plt.plot(t, ae_duke, color='r', alpha=0.8)
plt.title("Duke")
plt.ylim(-1, 1)
plt.subplot(3, 1, 3)
#librosa.display.waveshow(redhot, alpha=0.5)
plt.plot(t, ae_redhot, color='r', alpha=0.8)
plt.title("Red Hot Chili Peppers")
plt.ylim(-1, 1)
plt.subplots_adjust(hspace=0.5)
plt.show()
        
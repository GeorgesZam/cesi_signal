import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import simpleaudio as sa
def str_binaire(message):
    binare = ''.join(format(ord(char), '08b') for char in message)
    return binare


def binaire_str(binaire):
    # Diviser la chaîne binaire en octets de 8 bits
    bytes_array = [binaire[i:i + 8] for i in range(0, len(binaire), 8)]

    # Convertir chaque octet en caractère ASCII
    char_traduit = [chr(int(byte, 2)) for byte in bytes_array]

    # Joindre les caractères pour former la chaîne finale
    resultat = ''.join(char_traduit)
    return resultat


def tramer_binaire(binaire, key):
    """
    Ajoute un CRC, un bit de début, un bit de parité et un bit de fin à la séquence binaire.

    :param binaire: La séquence binaire originale.
    :param key: Clé CRC (polynôme).
    :return: La séquence binaire avec les bits de trame et CRC ajoutés.
    """
    # Calculer le CRC
    codeword, crc = encodeData(binaire, key)

    # Calculer le bit de parité (nombre de '1' est pair)
    bit_parite = '1' if codeword.count('1') % 2 == 0 else '0'

    # Ajouter bit de début (0), bit de parité et bit de fin (1) à la séquence binaire
    trame = '0' + codeword + bit_parite + '1'
    return trame, crc


def detramer_binaire(trame, key):
    """
    Enlève le bit de début, le bit de parité, le bit de fin et vérifie le CRC.

    :param trame: La séquence binaire avec les bits de trame.
    :param key: Clé CRC (polynôme).
    :return: La séquence binaire originale sans les bits de trame et le statut de vérification CRC.
    """
    # Enlever le bit de début et de fin
    codeword = trame[1:-2]

    # Vérifier le CRC
    crc_check = verifyData(codeword, key)

    # Enlever le CRC pour obtenir les données originales
    data = codeword[:-len(key)+1]

    return data, crc_check


def binary_to_manchester(binary_data):
    """Convertit une chaîne binaire en signal Manchester."""
    manchester_signal = ''
    for bit in binary_data:
        if bit == '0':
            manchester_signal += '01'
        elif bit == '1':
            manchester_signal += '10'
    return manchester_signal

def manchester_to_binary(manchester_signal):
    """Convertit un signal Manchester en chaîne binaire."""
    binary_data = ''
    i = 0
    while i < len(manchester_signal):
        if manchester_signal[i:i+2] == '01':
            binary_data += '0'
        elif manchester_signal[i:i+2] == '10':
            binary_data += '1'
        i += 2
    return binary_data


def fsk_demodulation(fsk_signal, Ns, Fe, A1, A2, fp1, fp2):
    # Générer le vecteur temps t1 pour les porteuses
    Duree1 = Ns / Fe
    t1 = np.arange(0, Duree1, 1 / Fe)

    # Générer les porteuses P1 et P2
    P1 = A1 * np.sin(2 * np.pi * fp1 * t1)
    P2 = A2 * np.sin(2 * np.pi * fp2 * t1)

    # Initialisation des vecteurs de démodulation
    demodulated_bits = []

    # Démodulation FSK
    for i in range(0, len(fsk_signal), len(P1)):
        segment = fsk_signal[i:i + len(P1)]

        correlation_P1 = np.abs(np.dot(segment, P1))
        correlation_P2 = np.abs(np.dot(segment, P2))

        if correlation_P1 > correlation_P2:
            demodulated_bits.append(1)
        else:
            demodulated_bits.append(0)

    return np.array(demodulated_bits)


def fsk_modulation(binary_data, Ns, Fe, A1, A2, fp1, fp2):
    """Modulation FSK d'une séquence binaire."""
    # Dupliquer chaque bit du message binaire Ns fois
    message_duplicated = np.repeat(binary_data, Ns,)

    # Générer le vecteur temps t1
    Duree1 = Ns / Fe
    t1 = np.arange(0, Duree1, 1 / Fe)

    # Générer le vecteur temps t
    Duree = len(message_duplicated) / Fe
    t = np.arange(0, Duree, 1 / Fe)

    # Génération des porteuses P1 pour le bit '1' et P2 pour le bit '0'
    P1 = A1 * np.sin(2 * np.pi * fp1 * t1)
    P2 = A2 * np.sin(2 * np.pi * fp2 * t1)

    # Réalisation de la modulation FSK
    fsk_signal = []
    for bit in message_duplicated:
        if bit == 1:
            fsk_signal.extend(list(P1))
        else:
            fsk_signal.extend(list(P2))

    return t, np.array(fsk_signal)


def non_repete(binaire, Ns):
    list_non_repete = []

    for i in range(0, len(binaire), Ns):
        if binaire[i] == 1 and binaire[i+1] == 1:
            list_non_repete.append(1)
        elif binaire[i] == 0 and binaire[i+1] == 1:
            list_non_repete.append(1)
        elif binaire[i] == 1 and binaire[i+1] == 0:
            list_non_repete.append(0)
        else:
            # Compléter avec le cas où la condition n'est pas satisfaite
            list_non_repete.append(binaire[i])

    return list_non_repete


def afficher_fsk_modulation_et_composantes(binary_data, Ns, Fe, A1, A2, fp1, fp2):
    # Dupliquer chaque bit du message binaire Ns fois
    message_duplicated = np.repeat(binary_data, Ns)

    # Générer le vecteur temps t pour le signal modulé
    Duree = len(message_duplicated) / Fe
    t = np.linspace(0, Duree, len(message_duplicated))

    # Générer les porteuses P1 et P2
    P1 = A1 * np.sin(2 * np.pi * fp1 * t)
    P2 = A2 * np.sin(2 * np.pi * fp2 * t)

    # Réalisation de la modulation FSK
    fsk_signal = A1 * np.sin(2 * np.pi * fp1 * t) * (message_duplicated == 1) + \
                 A2 * np.sin(2 * np.pi * fp2 * t) * (message_duplicated == 0)

    # Affichage
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Signal Binaire
    axs[0].step(t, message_duplicated, where='post', label='Signal Binaire')
    axs[0].set_ylim(-0.2, 1.2)
    axs[0].set_xlim(min(t), max(t))
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Signal Binaire')
    axs[0].grid(True)

    # Onde Porteuse
    axs[1].plot(t, P1, label='Porteuse pour 1', color='green')
    axs[1].plot(t, P2, label='Porteuse pour 0', color='red')
    axs[1].set_xlim(min(t), max(t))
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Ondes Porteuses')
    axs[1].grid(True)
    axs[1].legend()

    # Signal Modulé FSK
    axs[2].plot(t, fsk_signal, label='Signal Modulé FSK', color='blue')
    axs[2].set_xlim(min(t), max(t))
    axs[2].set_xlabel('Temps (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_title('Signal Modulé FSK')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def xor(a, b):
    """ Fonction XOR pour deux chaînes de la même longueur """
    result = []
    for i in range(1, len(b)):
        result.append('0' if a[i] == b[i] else '1')
    return ''.join(result)


def mod2div(dividend, divisor):
    """ Fonction pour effectuer la division modulo-2 """
    pick = len(divisor)
    tmp = dividend[0 : pick]

    while pick < len(dividend):
        if tmp[0] == '1':
            tmp = xor(divisor, tmp) + dividend[pick]
        else:
            tmp = xor('0'*pick, tmp) + dividend[pick]
        pick += 1

    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)

    return tmp


def encodeData(data, key):
    """ Fonction pour encoder les données en utilisant le CRC """
    l_key = len(key)
    appended_data = data + '0'*(l_key-1)
    remainder = mod2div(appended_data, key)
    codeword = data + remainder
    return codeword, remainder


def verifyData(CRC, key):
    """ Vérifie si les données codées avec CRC sont correctes """
    remainder = mod2div(CRC, key)
    return int(remainder) == 0


def wav_to_binary(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Lire les données du fichier WAV
        signal = wav_file.readframes(wav_file.getnframes())

    # Convertir les données en binaire
    binary_data = ''.join(format(byte, '08b') for byte in signal)
    return binary_data

def binary_to_wav(binary_data, file_path, sample_rate):
    # Convertir les données binaires en bytes
    byte_data = bytes(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))

    with wave.open(file_path, 'wb') as wav_file:
        # Paramètres du fichier WAV
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(1)  # Taille de l'échantillon en bytes
        wav_file.setframerate(sample_rate)

        # Écrire les données dans le fichier WAV
        wav_file.writeframes(byte_data)


def get_sample_rate(file_path):
    """
    Récupère la fréquence d'échantillonnage d'un fichier WAV.

    :param file_path: Chemin du fichier WAV.
    :return: Fréquence d'échantillonnage du fichier WAV.
    """
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
    return sample_rate


#prgramme principal
def main():
    # Clé CRC (polynôme)
    key = "1001"   #clé CRC


    #choisir quel type d'input

    choix = input("tapper 1 pour du texte, 2 pour un fichier audio")
    if choix == '1' :

        # Demander à l'utilisateur de taper un texte
        texte_utilisateur = input("Entrez votre texte : ")

        # Transformez le texte en binaire
        binaire_data = str_binaire(texte_utilisateur)
        print(f"Texte en binaire : {binaire_data}")

        # Tramer le binaire avant l'encodage Manchester
        binaire_trame, crc = tramer_binaire(binaire_data, key)
        print(f"Trame binaire avant modulation : {binaire_trame}")

        # Encodage Manchester de la trame binaire
        manchester_data = binary_to_manchester(binaire_trame)
        print(f"encodage en manchester : {manchester_data}")

        # Préparer les données pour la modulation FSK
        manchester_data_array = np.array([int(bit) for bit in manchester_data])

        # Modulation FSK
        t, fsk_signal = fsk_modulation(manchester_data_array, Ns=5, Fe=8000, A1=1, A2=1, fp1=25000, fp2=50000)

        # Afficher les composantes de la modulation FSK et le graphique
        afficher_fsk_modulation_et_composantes(manchester_data_array, Ns=5, Fe=8000, A1=1, A2=1, fp1=25000, fp2=50000)

        # Démodulation FSK
        demodulated_bits = non_repete(fsk_demodulation(fsk_signal, Ns=5, Fe=8000, A1=1, A2=1, fp1=25000, fp2=50000), Ns=5)

        # Décodage Manchester du signal démodulé
        manchester_decoded = manchester_to_binary(''.join(map(str, demodulated_bits)))
        print(f"Trame binaire après démodulation : {manchester_decoded}")

        # Détramer le binaire après le décodage Manchester
        binaire_detrame, crc_check = detramer_binaire(manchester_decoded, key)
        print(f"Texte binaire sans trame : {binaire_detrame}")

        # Convertir le binaire détrame en texte
        texte_decoded = binaire_str(binaire_detrame)
        print(f"Texte décodé : {texte_decoded}")


        # Vérifier l'intégrité des données
        donnees_intactes = crc_check

        if donnees_intactes:
            print("Aucune donnée n'est perdue. L'intégrité des données est confirmée.")
        else:
            print("Des données ont été perdues ou altérées pendant la modulation/démodulation.")

    elif choix == '2':

        input_wav = input("déposer le fichier wav sur votre bureau puis taper son nom")

        #utilise os pour obtenir le la localisation du fichier sur le burreau d'un (((mac)))
        input_wav_path = os.path.join(os.path.expanduser("~/Desktop"), input_wav)

        sample_rate = get_sample_rate(input_wav_path)
        print(f"Fréquence d'échantillonnage du fichier original: {sample_rate} Hz")

        binary_data = wav_to_binary(input_wav_path)
        # Tramer le binaire avant l'encodage Manchester
        binaire_trame, crc = tramer_binaire(binary_data, key)
        print(f"Trame binaire avant modulation : {binaire_trame}")

        # Encodage Manchester de la trame binaire
        manchester_data = binary_to_manchester(binaire_trame)

        # Préparer les données pour la modulation FSK
        manchester_data_array = np.array([int(bit) for bit in manchester_data])

        # Modulation FSK
        t, fsk_signal = fsk_modulation(manchester_data_array, Ns=5, Fe=8000, A1=1, A2=1, fp1=25000, fp2=50000)

        # Afficher les composantes de la modulation FSK et le graphique
        afficher_fsk_modulation_et_composantes(manchester_data_array, Ns=5, Fe=80, A1=1, A2=1, fp1=250, fp2=500)

        # Démodulation FSK
        demodulated_bits = non_repete(fsk_demodulation(fsk_signal, Ns=5, Fe=8000, A1=1, A2=1, fp1=25000, fp2=50000), Ns=5)
        print(f"donnée après demodulation :{demodulated_bits}")
        # Décodage Manchester du signal démodulé
        manchester_decoded = manchester_to_binary(''.join(map(str, demodulated_bits)))
        print(f" Binaire après démodulation : {manchester_decoded}")

        # Détramer le binaire après le décodage Manchester
        binaire_detrame, crc_check = detramer_binaire(manchester_decoded, key)
        print(f"Texte binaire sans trame : {binaire_detrame}")


        # Transformer en fichier WAV
        output_wav_path = os.path.join(os.path.expanduser("~/Desktop"), "output_final.wav")
        binary_to_wav(binaire_detrame, output_wav_path, 2200 )


        if crc_check:
            print("Aucune donnée n'est perdue. L'intégrité des données est confirmée.")
        else:
            print("Des données ont été perdues ou altérées pendant la modulation/démodulation.")


if __name__ == "__main__":
    main()

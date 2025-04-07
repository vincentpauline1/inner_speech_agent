# Main routine for HMM training/decoding
# T. Hueber - CNRS/GIPSA-lab - 2019
# MASTER2 SIGMA 
########################################
from __future__ import print_function
import numpy as np
import scipy
import scipy.io as sio
import os
from os import listdir, mkdir, system
from os.path import join, isdir, basename, splitext
import pdb # debugger
import glob
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import math

# CONFIG
#########
htk_bin_dir = "htk/bin/linux/"; #HTK binaries
audio_dir =  "data/imitative_agent_inner_speech/pb2007/overt/repeated/" ; #"data/wav16" # audio data "data/pb2007/_wav16/"
corpus_mlf = "data/pb2007_lab_aligned.mlf"; # "data/Thomas_lemonde_1_150_aligned.mlf"; ; # #phonetic segmentation of audio data 
# /mnt/c/Users/vpaul/OneDrive - CentraleSupelec/Inner_Speech/TP_HMM_Decoder/tp_reco_htk_sigma/data/imitative_agent/estimated/pb2007
train_test_ratio= 0.2 # train/test data partioning
    
mfcc_config_file = "htk/config/mfcc.cnf"; # acoustic feature extraction 
phonelist = "data/phonelist"; # HMM list (aka list of phonemes)
hmm_proto = "models/proto"; # HMM topology 

nIte = 10; # number of iteration of the EM alggorithm (Baum-welch)

# decoding options (viterbi decoding)
dict_filename = "lm/dict_phones";
grammar = "lm/grammar_phones";
wip = -0; # model insertion penalty

# Make some noise! 
noise_filename = "data/imitative_agent_inner_speech/pb2007/inner/repeated/item_0007.wav"; #"data/babble16.wav"

target_snr = 0; # in dB (warning, should be a float)

# Steps
step_add_noise_train = 0;
step_add_noise_test = 0;
step_extract_features = 1;
step_train = 1;
step_hmm_gmm = 1;
step_test = 1;

seed2 = 42 # for reproducibility

################
# SUBFUNCTIONS #
################
def do_extract_mfcc(all_audio_filenames, ind,target_scp_filename):
    if isdir("data/mfcc1/") is False:
        mkdir("data/mfcc1/")

    all_ind_audio_filenames = [all_audio_filenames[index] for index in ind]

    f = open(target_scp_filename,'w')
    for k in range(np.shape(all_ind_audio_filenames)[0]):
        audio_filename_no_root = basename(all_ind_audio_filenames[k])
        audio_basename = splitext(audio_filename_no_root)[0]
        f.write('data/mfcc1/' + audio_basename + '.mfcc\n')

        system(htk_bin_dir + '/HCopy -C ' + mfcc_config_file + ' ' +  all_ind_audio_filenames[k] + ' ' + 'data/mfcc1/' + audio_basename + '.mfcc');
        
    f.close()

############################################   

# def do_train():
#     # Computing variance floors
#     print("Computing variance of all features (variance flooring)\n");
#     if isdir("models/hmm_0")==False:
#         mkdir("models/hmm_0"); 

#     system(htk_bin_dir + "/HCompV -T 2 -f 0.01 -m -S " + "data/train.scp" + " -M models/hmm_0 -o models/average.mmf " + hmm_proto);
    
#     # Generating hmm template
#     system("head -n 1 " + hmm_proto + " > models/hmm_0/init.mmf");
#     system("cat models/hmm_0/vFloors >> models/hmm_0/init.mmf");

#     # HMM parameters estimation using Viterbi followed by Baum-welch (EM) alg.
#     all_phones = [line.rstrip() for line in open(phonelist)]

#     if isdir("models/hinit")==False:
#         mkdir("models/hinit/");

#     if isdir("models/hrest")==False:
#         mkdir("models/hrest/");

#     for p in range(np.shape(all_phones)[0]):
#         print("===============" + all_phones[p] + "================\n");
#         system(htk_bin_dir +"/HInit -T 000001 -A -H models/hmm_0/init.mmf -M models/hinit/ -I " + corpus_mlf + " -S " + "data/train.scp" + " -l "+ all_phones[p] + " -o " + all_phones[p] + " " + hmm_proto);
#         system(htk_bin_dir + "/HRest -A -T 000001 -H models/hmm_0/init.mmf -M models/hrest/ -I " + corpus_mlf + " -S " + "data/train.scp" + " -l " + all_phones[p]+ " models/hinit/" + all_phones[p]);
        
#     # Making monophone mmf
#     # load variance floor macro
#     f=open("htk/scripts/lvf.hed","w")
#     f.write("FV \"models/hmm_0/vFloors\"\n");
#     f.close();

#     if isdir("models/herest_0")==False:
#         mkdir("models/herest_0")

#     system(htk_bin_dir + "/HHEd -d models/hrest/ -w models/herest_0/monophone.mmf htk/scripts/lvf.hed " + phonelist);

#     # HMM parameter generation using embedded version of Baum-welch algorithm
#     for i in range(nIte):
#         if isdir("models/herest_" + str(i+1))==False:
#             mkdir("models/herest_" + str(i+1))
	
#         # embedded reestimation
#         system(htk_bin_dir + "/HERest -A -I " + corpus_mlf + " -S " + "data/train.scp" + " -H models/herest_" + str(i) + "/monophone.mmf -M models/herest_" + str(i+1) + " " + phonelist);

#         # Make a copy of last model parameters
#         system("cp models/herest_" + str(i+1) + "/monophone.mmf models/herest_" + str(i+1) + "/monophone_gmm.mmf")

#     if step_hmm_gmm:
#         # Increase number of gaussians per state up to 2
#         f = open("./hhed.cnf","w");
#         f.write("MU %i {*.state[2-4].mix}\n" % 2);
#         f.close()

#         system(htk_bin_dir + "/HHEd -A -H models/herest_" + str(i+1) + "/monophone_gmm.mmf ./hhed.cnf " + phonelist);
    
#         # Re-estimate model parameters (let's do 5 iterations of EM)
#         for r in range (5):
#             system(htk_bin_dir + "/HERest -T 0 -S " + "data/train.scp" + " -H models/herest_" + str(i+1) + "/monophone_gmm.mmf -I " + corpus_mlf + " " + phonelist)
        
#             # ... up to 4
#             f = open("./hhed.cnf","w");
#             f.write("MU %i {*.state[2-4].mix}\n" % 4);
#             f.close()

#             system(htk_bin_dir + "/HHEd -A -H models/herest_" + str(i+1) + "/monophone_gmm.mmf ./hhed.cnf " + phonelist);

#             # Again, we re-estimate model parameters (let's do 5 iterations of EM)
#             for r in range (5):
#                 system(htk_bin_dir + "/HERest -A -T 0 -S " + "data/train.scp" + " -H models/herest_" + str(i+1) + "/monophone_gmm.mmf -I " + corpus_mlf + " " + phonelist)   

################ Reduce /v/ weight #################################

import os
from os import system, mkdir
from os.path import isdir

# def do_train():
#     # Computing variance floors
#     print("Computing variance of all features (variance flooring)\n")
#     if not isdir("models/hmm_0"):
#         mkdir("models/hmm_0")

#     system(f"{htk_bin_dir}/HCompV -T 2 -f 0.01 -m -S data/train.scp -M models/hmm_0 -o models/average.mmf {hmm_proto}")

#     # Generating HMM template
#     system(f"head -n 1 {hmm_proto} > models/hmm_0/init.mmf")
#     system("cat models/hmm_0/vFloors >> models/hmm_0/init.mmf")

#     # HMM parameters estimation using Viterbi followed by Baum-Welch (EM) algorithm
#     with open(phonelist, 'r') as f:
#         all_phones = [line.strip() for line in f]

#     if not isdir("models/hinit"):
#         mkdir("models/hinit")

#     if not isdir("models/hrest"):
#         mkdir("models/hrest")

#     for phone in all_phones:
#         print(f"=============== {phone} ===============\n")
#         system(f"{htk_bin_dir}/HInit -T 000001 -A -H models/hmm_0/init.mmf -M models/hinit/ -I {corpus_mlf} -S data/train.scp -l {phone} -o {phone} {hmm_proto}")
#         system(f"{htk_bin_dir}/HRest -A -T 000001 -H models/hmm_0/init.mmf -M models/hrest/ -I {corpus_mlf} -S data/train.scp -l {phone} models/hinit/{phone}")

#     # Making monophone MMF
#     # Load variance floor macro
#     with open("htk/scripts/lvf.hed", "w") as f:
#         f.write('FV "models/hmm_0/vFloors"\n')

#     if not isdir("models/herest_0"):
#         mkdir("models/herest_0")

#     system(f"{htk_bin_dir}/HHEd -d models/hrest/ -w models/herest_0/monophone.mmf htk/scripts/lvf.hed {phonelist}")

#     # HMM parameter generation using embedded version of Baum-Welch algorithm
#     for i in range(nIte):
#         herest_dir_prev = f"models/herest_{i}"
#         herest_dir_next = f"models/herest_{i+1}"
#         if not isdir(herest_dir_next):
#             mkdir(herest_dir_next)

#         # Embedded re-estimation
#         system(f"{htk_bin_dir}/HERest -A -I {corpus_mlf} -S data/train.scp -H models/herest_{i}/monophone.mmf -M {herest_dir_next} {phonelist}")

#         # Make a copy of last model parameters
#         system(f"cp models/herest_{i+1}/monophone.mmf models/herest_{i+1}/monophone_gmm.mmf")

#     if step_hmm_gmm:
#         # Create a customized HHEd configuration to reduce /v/ weight
#         with open("./hhed.cnf", "w") as f:
#             # Assign fewer mixtures to /v/
#             f.write("MU 2 {*.state[2-4].mix}\n")  # Default for all phones
#             f.write("MU 1 {/v/.state[2-4].mix}\n")  # Reduced mixtures for /v/
        
#         system(f"{htk_bin_dir}/HHEd -A -H models/herest_{i+1}/monophone_gmm.mmf ./hhed.cnf {phonelist}")

#         # Re-estimate model parameters with the new configuration
#         for r in range(5):
#             system(f"{htk_bin_dir}/HERest -A -T 0 -S data/train.scp -H models/herest_{i+1}/monophone_gmm.mmf -I {corpus_mlf} {phonelist}")

#             # Update HHEd configuration to increase mixtures for other phones, keeping /v/ at 1
#             with open("./hhed.cnf", "w") as f:
#                 f.write("MU 4 {*.state[2-4].mix}\n")  # Increase mixtures for other phones
#                 f.write("MU 1 {/v/.state[2-4].mix}\n")  # Keep /v/ mixtures low

#             system(f"{htk_bin_dir}/HHEd -A -H models/herest_{i+1}/monophone_gmm.mmf ./hhed.cnf {phonelist}")

#             # Re-estimate model parameters again
#             for _ in range(5):
#                 system(f"{htk_bin_dir}/HERest -A -T 0 -S data/train.scp -H models/herest_{i+1}/monophone_gmm.mmf -I {corpus_mlf} {phonelist}")





def do_train():
    # Computing variance floors
    print("Computing variance of all features (variance flooring)\n");
    if isdir("models/hmm_0")==False:
        mkdir("models/hmm_0"); 

    system(htk_bin_dir + "/HCompV -T 2 -f 0.01 -m -S " + "data/train.scp" + " -M models/hmm_0 -o models/average.mmf " + hmm_proto);
    
    # Generating hmm template
    system("head -n 1 " + hmm_proto + " > models/hmm_0/init.mmf");
    system("cat models/hmm_0/vFloors >> models/hmm_0/init.mmf");

    # HMM parameters estimation using Viterbi followed by Baum-welch (EM) alg.
    all_phones = [line.rstrip() for line in open(phonelist)]

    if isdir("models/hinit")==False:
        mkdir("models/hinit/");

    if isdir("models/hrest")==False:
        mkdir("models/hrest/");

    for p in range(np.shape(all_phones)[0]):
	
        print("===============" + all_phones[p] + "================\n");
        system(htk_bin_dir +"/HInit -T 000001 -A -H models/hmm_0/init.mmf -M models/hinit/ -I " + corpus_mlf + " -S " + "data/train.scp" + " -l "+ all_phones[p] + " -o " + all_phones[p] + " " + hmm_proto)

        system(htk_bin_dir + "/HRest -A -T 000001 -H models/hmm_0/init.mmf -M models/hrest/ -I " + corpus_mlf + " -S " + "data/train.scp" + " -l " + all_phones[p]+ " models/hinit/" + all_phones[p])
        # pdb.set_trace()

    # Making monophone mmf
    # load variance floor macro
    f=open("htk/scripts/lvf.hed","w")
    f.write("FV \"models/hmm_0/vFloors\"\n");
    f.close();

    if isdir("models/herest_0")==False:
        mkdir("models/herest_0")

    system(htk_bin_dir + "/HHEd -d models/hrest/ -w models/herest_0/monophone.mmf htk/scripts/lvf.hed " + phonelist);

    # HMM parameter generation using embedded version of Baum-welch algorithm
    for i in range(nIte):
        if isdir("models/herest_" + str(i+1))==False:
            mkdir("models/herest_" + str(i+1))
	
    # embedded reestimation
    system(htk_bin_dir + "/HERest -t 120.0 60.0 240.0 -A -I " + corpus_mlf + " -S " + "data/train.scp" + " -H models/herest_" + str(i) + "/monophone.mmf -M models/herest_" + str(i+1) + " " + phonelist +" -T 00001 -z");

    # Make a copy of last model parameters
    system("cp models/herest_" + str(i+1) + "/monophone.mmf models/herest_" + str(i+1) + "/monophone_gmm.mmf")

    if step_hmm_gmm:
        # Increase number of gaussians per state up to 2
        f = open("./hhed.cnf","w");
        f.write("MU %i {*.state[2-4].mix}\n" % 2);
        f.close()

        system(htk_bin_dir + "/HHEd -A -H models/herest_" + str(i+1) + "/monophone_gmm.mmf ./hhed.cnf " + phonelist);
    
        # Re-estimate model parameters (let's do 5 iterations of EM)
        for r in range (5):
            system(htk_bin_dir + "/HERest -T 0 -S " + "data/train.scp" + " -H models/herest_" + str(i+1) + "/monophone_gmm.mmf -I " + corpus_mlf + " " + phonelist)
        
            # ... up to 4
            f = open("./hhed.cnf","w");
            f.write("MU %i {*.state[2-4].mix}\n" % 4);
            f.close()

            system(htk_bin_dir + "/HHEd -A -H models/herest_" + str(i+1) + "/monophone_gmm.mmf ./hhed.cnf " + phonelist);

            # Again, we re-estimate model parameters (let's do 5 iterations of EM)
            for r in range (5):
                system(htk_bin_dir + "/HERest -A -T 0 -S " + "data/train.scp" + " -H models/herest_" + str(i+1) + "/monophone_gmm.mmf -I " + corpus_mlf + " " + phonelist)    
    
############################################
def do_test():
    # convert grammar rules to decoding network
    system(htk_bin_dir + "/HParse -A " + grammar + " lm/wnet")

    # phonetic decoding using Viterbi algorithm (Token Passing)
    system(htk_bin_dir + "/HVite -A -y rec -p " + str(0.0) + " -m -T 1 -S " + "data/test1.scp" + " -H models/herest_" + str(nIte) + "/monophone_gmm.mmf -i data/rec.mlf -w lm/wnet " + dict_filename + " " + phonelist)

    # Calculate WER    str(wip) 
    x = system(htk_bin_dir + "/HResults -A -X lab -s -t -f -p -I " + corpus_mlf + " " + phonelist + " data/rec.mlf")
    print(f'the HResult return is {x}')

##############################################

# def add_noise(all_audio_filenames,audio_dir_noise,noise_filename,target_snr):
#     if isdir(audio_dir_noise) is False:
#         mkdir(audio_dir_noise)

#     y_noise,sr_noise = librosa.load(noise_filename, sr=None)
#     #all_ind_audio_filenames = [all_audio_filenames[index] for index in ind]

#     for k in range(np.shape(all_audio_filenames)[0]):
#         audio_filename_no_root = basename(all_audio_filenames[k])
#         audio_basename = splitext(audio_filename_no_root)[0]
        
#         y, sr = librosa.load(all_audio_filenames[k],sr=None)
#         rms_y = math.sqrt(np.mean(y**2))

#         noise_part = y_noise[range(np.shape(y)[0]-1)] 
#         rms_noise = math.sqrt(np.mean(noise_part**2))
#         target_rms_noise = np.sqrt((rms_y**2)/(10**(target_snr/10)))

#         #current_snr = 10*np.log10((rms_y**2)/(rms_noise**2))
#         #print("(initial) RMS_noise=%.2f, RMS_signal=%.2f, SNR = %.2f\n, RMS_noise_target=%.2f" % (rms_noise,rms_y,current_snr,target_rms_noise));
        
#         while rms_noise>target_rms_noise :
#             noise_part = 0.99*noise_part 
#             rms_noise = math.sqrt(np.mean(noise_part**2))
#             current_snr = 10*np.log10((rms_y**2)/(rms_noise**2))
#             #print("RMS_noise=%.3f, RMS_signal=%.3f, SNR = %.3f\n" % (rms_noise,rms_y,current_snr));
 
#         y_plus_noise = y + noise_part;
#         current_snr = 10*np.log10((rms_y**2)/(rms_noise**2))
#         #print("(final) RMS_noise=%.2f, RMS_signal=%.2f, SNR = %.2f\n" % (rms_noise,rms_y,current_snr));
#         sf.write(audio_dir_noise + '/'+ audio_basename + '.wav',y_plus_noise,sr);


def add_noise(all_audio_filenames, audio_dir_noise, noise_filename, target_snr):
    """
    Adds noise to a list of audio files at a specified Signal-to-Noise Ratio (SNR).

    Parameters:
    - all_audio_filenames (list of str): List of paths to the audio files to be noised.
    - audio_dir_noise (str): Directory where the noised audio files will be saved.
    - noise_filename (str): Path to the noise audio file.
    - target_snr (float): Desired SNR in decibels (dB).

    Returns:
    - None
    """
    # Create the noise directory if it doesn't exist
    if not isdir(audio_dir_noise):
        mkdir(audio_dir_noise)
        print(f"Created directory: {audio_dir_noise}")

    # Load the noise audio
    try:
        y_noise, sr_noise = librosa.load(noise_filename, sr=None)
        print(f"Loaded noise file '{noise_filename}' with sampling rate {sr_noise} Hz.")
    except Exception as e:
        print(f"Error loading noise file '{noise_filename}': {e}")
        return

    # Iterate over each audio file
    for idx, audio_path in enumerate(all_audio_filenames, 1):
        audio_filename_no_root = basename(audio_path)
        audio_basename = splitext(audio_filename_no_root)[0]
        output_path = os.path.join(audio_dir_noise, f"{audio_basename}.wav")

        # Load the target audio file
        try:
            y, sr = librosa.load(audio_path, sr=None)
            print(f"[{idx}/{len(all_audio_filenames)}] Loaded '{audio_filename_no_root}' with sampling rate {sr} Hz.")
        except Exception as e:
            print(f"Error loading audio file '{audio_path}': {e}")
            continue  # Skip to the next file

        # Ensure the sampling rates match
        if sr != sr_noise:
            print(f"Sampling rate mismatch for '{audio_filename_no_root}': audio SR={sr} vs noise SR={sr_noise}. Resampling noise.")
            try:
                y_noise_resampled = librosa.resample(y_noise, orig_sr=sr_noise, target_sr=sr)
                y_noise = y_noise_resampled
                sr_noise = sr  # Update the sampling rate after resampling
                print(f"Resampled noise to {sr} Hz.")
            except Exception as e:
                print(f"Error resampling noise for '{audio_filename_no_root}': {e}")
                continue  # Skip to the next file

        # Calculate RMS of the audio signal
        rms_y = math.sqrt(np.mean(y ** 2))
        if rms_y == 0:
            print(f"Warning: Audio file '{audio_filename_no_root}' has zero energy. Skipping noise addition.")
            sf.write(output_path, y, sr)
            continue

        # Prepare the noise segment
        audio_length = len(y)
        noise_length = len(y_noise)

        if noise_length < audio_length:
            # Repeat the noise to match the length of the audio
            repeats = int(np.ceil(audio_length / noise_length))
            y_noise_extended = np.tile(y_noise, repeats)
            noise_part = y_noise_extended[:audio_length]
            print(f"Extended noise from {noise_length} to {len(noise_part)} samples.")
        else:
            noise_part = y_noise[:audio_length]

        # Calculate RMS of the noise segment
        rms_noise = math.sqrt(np.mean(noise_part ** 2))
        if rms_noise == 0:
            print(f"Warning: Noise file '{noise_filename}' has zero energy. Skipping noise addition for '{audio_filename_no_root}'.")
            sf.write(output_path, y, sr)
            continue

        # Calculate the target RMS for the noise to achieve the desired SNR
        target_rms_noise = rms_y / (10 ** (target_snr / 20))

        # Scale the noise to match the target RMS
        scaling_factor = target_rms_noise / (rms_noise + 1e-8)  # Add epsilon to prevent division by zero
        noise_part_scaled = noise_part * scaling_factor
        rms_noise_scaled = math.sqrt(np.mean(noise_part_scaled ** 2))
        achieved_snr = 20 * math.log10(rms_y / (rms_noise_scaled + 1e-8))

        # Add the scaled noise to the original audio
        y_noisy = y + noise_part_scaled

        # Optionally, clip the values to avoid clipping in the audio signal
        max_val = max(np.max(np.abs(y_noisy)), 1.0)
        y_noisy = y_noisy / max_val

        # Save the noised audio
        try:
            sf.write(output_path, y_noisy, sr)
            print(f"Saved noised file '{output_path}' with achieved SNR: {achieved_snr:.2f} dB.")
        except Exception as e:
            print(f"Error saving noised file '{output_path}': {e}")
            continue  # Skip to the next file


############################################
def HTKWrite(fname, data, sampPeriod):
	nSamples = np.shape(data)[0]
	nFeatures = np.shape(data)[1]

	sampSize = nFeatures * 4
	paramKind = 9  # USER
	
	f = open(fname, 'wb')
	
	# Write header
	f.write(struct.pack('i', nSamples))
	f.write(struct.pack('i', sampPeriod))
	f.write(struct.pack('h', sampSize))
	f.write(struct.pack('h', paramKind))

	# Write data
	data.astype(float32).tofile(f)
	
	f.close()

############################################
def HTKRead(fname):
    f = open(fname, 'rb')
    
    nSamples = np.fromfile(f, 'i', 1, "")
    sampPeriod = np.fromfile(f, 'i', 1, "")  
    sampSize = np.fromfile(f, 'h', 1, "") 
    paramKind = np.fromfile(f, 'h',1, "") 

    nFeatures = sampSize/4

    # Read data
    data_vect  = np.fromfile(f,'f',nFeatures*nSamples,"") 
    data = data_vect.reshape((nSamples, nFeatures))
    f.close()
    
    return data

#######################

#########
# MAIN  #
#########
if __name__ == '__main__':

    # Split dataset into train and test set
    all_audio_filenames = glob.glob(audio_dir + '/*.wav')
    train_ind, test_ind = train_test_split(range(np.shape(all_audio_filenames)[0]), test_size = train_test_ratio, random_state=seed2)
    # test_ind = range(np.shape(all_audio_filenames)[0])
    # train_ind = []
    if step_add_noise_train | step_add_noise_test:
        print("Adding babble noise to clean audio ...\t");
        add_noise(all_audio_filenames,'data/wav16_noise',noise_filename,target_snr);
        print("Done\n");

    if step_extract_features:
        if step_add_noise_train:
            print("Extracting MFCC (noisy train set) ...\t");
            do_extract_mfcc(glob.glob('data/wav16_noise/*.wav'), train_ind, 'data/train1.scp');
            print("Done\n");
        else: 
            print("Extracting MFCC (train set) ...\t");
            do_extract_mfcc(all_audio_filenames, train_ind, 'data/train1.scp');
            print("Done\n");

        if step_add_noise_test:
            print("Extracting MFCC (noisy test set) ...\t");
            do_extract_mfcc(glob.glob('data/wav16_noise/*.wav'), test_ind, 'data/test1.scp');
            print("Done\n");
        else:
            print("Extracting MFCC (test set) ...\t");
            do_extract_mfcc(all_audio_filenames, test_ind, 'data/test1.scp');
            print("Done\n");
        
    if step_train:
        do_train();

    if step_test:
        do_test();

## END OF FILE
##############

    

# Action Annotations for EGTEA Gaze+

This file contains action annotations for EGTEA Gaze+ dataset 

* Each action label has the format of verb + (noun_1, noun_2, ...)
* All action labels and their IDs can be found in action_idx.txt
* Similarly, all verb and noun labels and their IDS are in verb_idx.txt and noun_idx.txt, respectively
* We include three different train/test splits for action recognition as train/test_split(1-3).txt
	* Each split contains its own train/test set (mutually exclusive)
	* Each split (train/test) file follows the format of VideoName ActionID VerbID (NounID_1, NounID_2)
	* VideoName is the file name of the cropped clips (Please download them separately). 
	* ActionID is used for action recognition. These IDs are the class index of actions. Please see action_idx.txt for the full action labels.
	* We encourage reporting results as the mean accuracy averaged across all classes across all three splits.

We also include the raw exported annotation file (CSV files) in the sub-folder of "raw_annotations". They are primarily used for book-keeping. If you are interested, please refer to their headers for the formats.

# Notes
* We have identified an accidental bias in the original first split, which has a much higher rate of clips from the same video. 
* We have thus re-sampled the first split and double check the statistics across splits.
* We also removed 4 action instances that have empty videos (frames been removed during de-identification).

Contact: yli440@gatech.edu
Last updated: Mar 6th, 2018
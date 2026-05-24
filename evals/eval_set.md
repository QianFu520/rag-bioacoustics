# RAG Project — Evaluation Test Set

**Corpus:** Three bioacoustics papers
- neuroethology_review.docx
- opensoundscape_methods.docx
- audiomoth_deployment.docx

**Categories:**
1. Single-fact lookup
2. Multi-chunk synthesis (within one paper)
3. Cross-document
4. Specific numerical / technical detail
5. Out-of-corpus / refusal

---
## Q1 — Category 1 (single-fact lookup)

**Paper:** neuroethology_review.docx

**Question:** In the Python package `noisereduce`, how is the background noise estimate computed for non-stationary noise reduction?

**Expected answer:** Using a time-smoothed spectrogram (computed with a forward and backward IIR filter), on a timescale parameterized by the expected signal length. Motivated by the Per-Channel Energy Normalization algorithm.

**Anchor:** "In the Python package noisereduce, this background estimate is computed using a time-smoothed spectrogram (using a forward and backward IIR filter) on a timescale parameterized by the expected signal length, an approach motivated by the Per-Channel Energy Normalization algorithm."

---
## Q2 — Category 1 (single-fact lookup)

**Paper:** neuroethology_review.docx

**Question:** What are the three related tasks that automatic vocalization annotation can be broken down into, and how is each defined?

**Expected answer:** Identification (what animal is vocalizing and at what times and frequency channels), segmentation (segmenting vocalizations into their constituent units), and labeling (grouping units into discrete element categories).

**Anchor:** "Automatic vocalization annotation can be broken down into three related tasks: identification, segmentation, labeling. Identifying refers to what animal is vocalizing and at what times and frequency channels. Segmentation refers to the segmentation of vocalizations into their constituent units, labeling then refers to grouping units into discrete element categories."

---
## Q3 — Category 1 (single-fact lookup)

**Paper:** opensoundscape_methods.docx

**Question:** What does the BoxedAnnotations class in OpenSoundscape do?

**Expected answer:** It is used to view and manipulate audio annotations; prepare annotated data for training and evaluation of automated classification algorithms; load and save annotation files; and filter, aggregate, manipulate and correct labels across a dataset. It also provides integration with Raven Pro and Raven Lite annotation software through the import and export of Raven-formatted files.

**Anchor:** "BoxedAnnotations class: View and manipulate audio annotations; prepare annotated data for training and evaluation of automated classification algorithms; load and save annotation files; filter, aggregate, manipulate and correct labels across a dataset."

---
## Q4 — Category 1 (single-fact lookup)

**Paper:** opensoundscape_methods.docx

**Question:** In Notebook 1's analysis of *Atelopus varius* (the variable harlequin frog) vocalizations, what produces the call's rapid amplitude modulation?

**Expected answer:** Constructive and destructive interference of tones differing by 130 Hz.

**Anchor:** "Inspecting the temporal and harmonic characteristics of the call leads to the insight that the call's rapid amplitude modulation is produced by constructive and destructive interference of tones differing by 130 Hz."

---
## Q5 — Category 1 (single-fact lookup)

**Paper:** audiomoth_deployment.docx

**Question:** Why is the Raspberry Pi described as a power-hungry choice for acoustic monitoring devices, and what does that imply for long-term deployments?

**Expected answer:** The Raspberry Pi typically runs a full Linux operating system, with power consumption varying from 80 mA to 260 mA depending on the model and use. This means long-term deployments require higher-capacity power sources such as 12-volt automotive batteries, which constrains the ability of these devices to provide substantial coverage on limited budgets.

**Anchor:** "The Raspberry Pi lends itself to this application since it is easy to expand, program, and configure. However, it typically runs a full Linux operating system, and thus is also relatively power-hungry, with consumption varying from 80 mA to 260 mA depending on the model and use. Long-term deployments consequently require higher-capacity power sources such as 12-volt automotive batteries, constraining the ability of these devices to provide substantial coverage on limited budgets."

---
## Q6 — Category 1 (single-fact lookup)

**Paper:** audiomoth_deployment.docx

**Question:** If an AudioMoth were to record bat echolocation continuously instead of using a detection algorithm, what would its nightly storage requirement and battery life be?

**Expected answer:** It would require 18 GB of storage each night and would exhaust three AAA-cell batteries in approximately three nights.

**Anchor:** "If each device recorded continuously, rather than using a detection algorithm, it would require 18 GB of storage each night and would exhaust the energy supply of three AAA-cell batteries in approximately three nights."

---
## Q7 — Category 2 (multi-chunk synthesis, single paper)

**Paper:** neuroethology_review.docx

**Question:** What is the difference between stationary and non-stationary noise reduction, and what specific approach does the Python package `noisereduce` use to handle non-stationary noise?

**Expected answer:** Stationary noise reduction handles noise that's constant in intensity and spectral shape over time (e.g., the hum of electronics), while non-stationary noise reduction targets noise that fluctuates in time (e.g., a plane flying overhead). The `noisereduce` package handles non-stationary noise by extending spectral gating: it computes a variable gate based on a current background-noise estimate, where that estimate is computed using a time-smoothed spectrogram (forward and backward IIR filter) on a timescale parameterized by the expected signal length, motivated by Per-Channel Energy Normalization.

**Anchor 1 (Section 2.1):** "Stationary noise reduction acts on noise that is stationary in intensity and spectral shape over time, such as the constant hum of electronics. Non-stationary noise reduction targets background noise that is non-stationary and can fluctuate in time, like the on-and-off presence of a plane flying overhead."

**Anchor 2 (Section 2.2):** "Spectral gating can be extended to non-stationary noise reduction by computing a variable gate based upon the current estimate of background noise. In the Python package noisereduce, this background estimate is computed using a time-smoothed spectrogram (using a forward and backward IIR filter) on a timescale parameterized by the expected signal length, an approach motivated by the Per-Channel Energy Normalization algorithm."

---

## Q8 — Category 2 (multi-chunk synthesis, single paper)

**Paper:** neuroethology_review.docx

**Question:** What is the distinction between local and global dimensionality reduction algorithms, and which type does UMAP fall into?

**Expected answer:** Global dimensionality reduction algorithms try to preserve every pairwise relationship in the dataset, while local dimensionality reduction algorithms emphasize preserving only relationships to nearby points (more similar items) in dataspace. UMAP is a local dimensionality reduction algorithm — alongside t-SNE, it is one of the two dominant local algorithms. Both operate by first computing a nearest-neighbors graph of pairwise relationships, then embedding that graph via gradient descent.

**Anchor 1 (Section 5.2):** "Algorithms that attempt to preserve every pairwise relationship are called global dimensionality reduction algorithms, while algorithms that emphasize capturing relationships only to nearby points in dataspace (more similar vocalizations) are called local dimensionality reduction algorithms."

**Anchor 2 (Section 5.2):** "At present, the two dominant local dimensionality reduction algorithms are UMAP and t-SNE. UMAP and t-SNE differ in several important ways beyond the scope of this paper, but their key intuition and the steps underlying the algorithms remain similar: first, compute a (nearest-neighbors) graph of pairwise relationships between nearest neighbors in the original dataset (e.g., using Euclidean distance or an arbitrary similarity metric) then, embed that graph into an embedding space via gradient descent."

---
## Q9 — Category 2 (multi-chunk synthesis, single paper)

**Paper:** opensoundscape_methods.docx

**Question:** What kinds of automated detection methods does OpenSoundscape support, and in what situation might signal processing methods be preferable to deep learning?

**Expected answer:** OpenSoundscape supports both machine learning (specifically deep learning with CNNs) and signal processing approaches for automated detection. Signal processing methods may be preferable to deep learning when little to no training data is available — a common situation in bioacoustics. Signal processing methods also offer interpretable parameters that can be tuned to biologically relevant values, and they are especially effective for detecting songs with stereotyped temporal structure, such as many anuran vocalizations and invertebrate stridulations.

**Anchor 1 (Section 3):** "The primary functionality of OpenSoundscape is the development and application of automated algorithms for locating biological sounds of interest in space and time. In bioacoustics, some automated recognition tasks are well suited to machine learning while others are best solved with signal processing, and OpenSoundscape provides functionality for both approaches."

**Anchor 2 (Section 4.3):** "Although deep learning models are popular and effective approaches to many automated detection problems, they can be difficult to apply when little to no training data is available. In these scenarios, which may be common for bioacoustics tasks, signal processing methods may be preferable. Unlike deep learning approaches, these methods have interpretable parameters that can be tuned to biologically relevant values by the user. They can be especially effective in detecting songs with stereotyped temporal structure, a feature of many anuran vocalizations and invertebrate stridulations."

---

## Q10 — Category 2 (multi-chunk synthesis, single paper)

**Paper:** opensoundscape_methods.docx

**Question:** What capability of OpenSoundscape's Audio and Spectrogram classes distinguishes them from other Python audio tools, and how is this capability demonstrated in the Notebook 1 case study?

**Expected answer:** OpenSoundscape's Audio and Spectrogram classes retain, manipulate, and inspect audio metadata and parameters alongside the raw sample data — a capability that increases interpretability and reproducibility during acoustic data analysis, and which (according to the paper) other available Python tools lack. Notebook 1 demonstrates this capability by using the Audio and Spectrogram classes to inspect a vocalization of *Atelopus varius* (the variable harlequin frog). By changing spectrogram parameters, the analysis reveals that the call's rapid amplitude modulation is produced by constructive and destructive interference of tones differing by 130 Hz.

**Anchor 1 (Section 3):** "OpenSoundscape also provides intuitive and robust Audio and Spectrogram classes for interacting with audio data. The ability to retain, manipulate and inspect attributes and metadata of audio files alongside the raw sample data increases interpretability and reproducibility during acoustic data analyses, but to our knowledge, all other available Python tools lack this functionality."

**Anchor 2 (Section 4.1):** "Notebook 1 uses the Audio and Spectrogram classes to inspect the characteristics of a vocalization of Atelopus varius (the variable harlequin frog), demonstrating the power of changing spectrogram parameters for revealing acoustic structure. Inspecting the temporal and harmonic characteristics of the call leads to the insight that the call's rapid amplitude modulation is produced by constructive and destructive interference of tones differing by 130 Hz."

---
## Q11 — Category 2 (multi-chunk synthesis, single paper)

**Paper:** audiomoth_deployment.docx

**Question:** What is the Goertzel algorithm and why is it used in AudioMoth's detection algorithms, and what algorithmic improvement does it offer over a Discrete Fourier Transform?

**Expected answer:** The Goertzel algorithm is a method for extracting specific frequency components from audio samples by evaluating a subset of terms of a full FFT, rather than computing the entire transform. It is used in AudioMoth's detection algorithms because frequency component extraction is a step common to all of them, and the Goertzel algorithm avoids the computational expense of running complete fast Fourier transforms. In terms of computational complexity, the Goertzel algorithm on a window of L samples is O(L), whereas a DFT on the same window is O(L log L) — a meaningful efficiency gain on low-power hardware like AudioMoth.

**Anchor 1 (Section 3.1):** "A step common to all acoustic detection algorithms is the extraction of features which define the target sound. The presence or absence of certain frequency components in a sound is one such feature. In order to perform frequency component extraction without the computational expense of frequently running complete fast Fourier transforms (FFT), the Goertzel algorithm is used to evaluate a subset of terms of a full FFT."

**Anchor 2 (Section 3.1):** "Calculating the magnitude of a target frequency within a set of samples in this way is computationally less complex than a discrete Fourier transform (DFT). With L samples in a window, the computational complexity of performing the Goertzel algorithm on that window is O(L), whereas a DFT on the same sample count is calculated to be O(L log L). The number of samples in a window is either 128 or 256 in each of the three algorithms described in this paper."

---

## Q12 — Category 2 (multi-chunk synthesis, single paper)

**Paper:** audiomoth_deployment.docx

**Question:** How did the performance of the gunshot detection algorithm evolve across its iterations, including how the first deployment informed the redesign?

**Expected answer:** The first iteration of the gunshot detection algorithm was deployed in Pook's Hill Reserve and achieved a true-positive rate of approximately 0.57 on gunshots fired at ranges up to 800 m through dense, broadleaf rainforest. The recordings collected during this deployment were used to reconfigure the model — forming new emission and transition probabilities, and producing a test dataset for future development. Subsequent iterations introduced a dedicated noise state to the HMM (instead of relying on the silence state to absorb false positives) and used updated model distributions based on the new recordings. The improved iteration, deployed in the Tapir Mountain Nature Reserve, achieved a true-positive rate of 0.84 on gunshots fired within 500 m of the AudioMoth.

**Anchor 1 (Section 3.4.6):** "On gunshots fired at ranges up to 800 m through dense, broadleaf rainforest, the detection algorithm responded to and recorded gunshots with a true positive rate of approximately 0.57. With the recordings collected during this deployment, the model was reconfigured, using the samples to form a new set of emission and transition probabilities as well as produce the test dataset to benchmark future developments of the algorithm."

**Anchor 2 (Section 3.4.6):** "Subsequent iterations of the algorithm also introduced the noise state to the HMM to accurately classify more instances of false positives, rather than attempt to encompass them within the silence state's probability distributions. An iteration with a noise state and updated model distributions based on the new recordings was developed and tested in early 2018, before being deployed in the final monitoring locations in the Tapir Mountain Nature Reserve, adjacent to the Pook's Hill Reserve. This iteration of the model was tested in a wider array of terrain, measuring the detection accuracy with hills and valleys between the source gunshot and the test devices. For gunshots fired within 500 m of the AudioMoth, the algorithm achieved a true-positive rate of 0.84."

---
## Q13 — Category 3 (cross-document synthesis)

**Papers:** opensoundscape_methods.docx + audiomoth_deployment.docx

**Question:** What kinds of automated recording units (ARUs) or acoustic monitoring hardware does the bioacoustics literature describe, and what are the main tradeoffs between commercial devices and open-source alternatives?

**Expected answer:** Automated recording units (ARUs) are hardware devices that enable researchers to collect landscape-scale acoustic data for biodiversity monitoring. The bioacoustics literature describes two main classes: commercial devices and open-source alternatives. Commercial devices such as the Wildlife Acoustics Song Meter series achieve high sound fidelity but cost US $800–1000 per unit, which limits research scope and adoption in developing countries. They are also closed-source, making them difficult to adapt to specific applications. Open-source alternatives address these tradeoffs by offering lower cost and the flexibility to modify hardware and software for specific applications.

**Anchor 1 (opensoundscape_methods, Section 1):** "Large-scale bioacoustic monitoring projects have become a popular approach to biodiversity monitoring. Effective and affordable automated recording unit (ARU) hardware enables researchers to collect landscape-scale acoustic data."

**Anchor 2 (audiomoth_deployment, Section 1):** "Currently, the majority of research projects in ecology and conservation using acoustics report using commercial devices such as the Wildlife Acoustics Song Meter series. Such devices achieve high sound fidelity at the cost of a high purchase price (US $800–1000 per unit). The high prices reflect the high quality of their microphones and the relatively small commercial market that they serve. However, not all applications require such fidelity. In these cases, the costs limit the scope and impact of the research, as well as their adoption in developing countries. Commercial monitoring devices are also closed-source, which makes it difficult or impossible to adapt their hardware or software to a specific application."

---
## Q14 — Category 3 (cross-document synthesis)

**Papers:** opensoundscape_methods.docx + audiomoth_deployment.docx

**Question:** How do the bioacoustics papers describe the use of convolutional neural networks (CNNs) for species or sound classification, and what are the practical constraints when deploying CNNs on low-cost hardware?

**Expected answer:** CNNs are described in the bioacoustics literature as a powerful approach for species or sound classification. The OpenSoundscape package provides high-level CNN tooling by interfacing with PyTorch, offering dozens of model architecture and training parameter options without requiring users to program in PyTorch directly, plus integration with Weights and Biases for monitoring training and inference. When deploying CNNs on low-cost hardware like AudioMoth (which has only 256 KB of flash memory), model size becomes a major practical constraint. To address this, depthwise separable convolution can be used in place of standard 3D convolution, which reduces both computational complexity and overall network size by convolving each channel separately and then combining them with a pointwise convolution.

**Anchor 1 (opensoundscape_methods, Section 3):** "OpenSoundscape interfaces with the popular PyTorch library to provide machine learning tools. Dozens of options for CNN model architectures and training parameters allow for easy and flexible model customization without requiring the user to program in PyTorch or other lower level packages. Furthermore, OpenSoundscape includes integration with the Weights and Biases platform to provide real-time monitoring of preprocessed samples, classification performance metrics and computational metering while training and predicting with machine learning models."

**Anchor 2 (audiomoth_deployment, Section 3.5.1):** "Model size is an important factor to consider when deploying a neural network on hardware with limited storage capacity, such as AudioMoth (256 KB of flash memory). To reduce the size and complexity of the CNN, depthwise separable convolution was used in place of standard 3D convolution. This is a common method of creating compact CNNs for image classification and involves convolving each RGB colour channel of the image separately, then using a pointwise convolution to combine them again. Doing so is more efficient than standard 3D convolution in terms of the number of operations and the number of parameters, thus reducing both the computational complexity and the overall size of the network."

---
## Q15 — Category 3 (cross-document synthesis)

**Papers:** neuroethology_review.docx + audiomoth_deployment.docx

**Question:** How are Hidden Markov Models (HMMs) discussed in the bioacoustics literature — what role do they play in analyzing animal vocalizations versus detecting non-biological acoustic events?

**Expected answer:** HMMs are discussed in two different bioacoustics contexts. In animal vocalization analysis (neuroethology), they are used to capture short-timescale sequential dynamics of vocal communication — specifically, to model the probability of transitions between vocal elements in birdsong. Hidden Markov Models, alongside related approaches like Probabilistic Suffix Trees, are used because they can compute more succinct high-order Markov relationships than plain higher-order Markov models, though data requirements are still a limiting factor. In the environmental monitoring context (AudioMoth gunshot detection), HMMs are used for non-vocalization acoustic event detection: the algorithm uses an HMM with three frequency components as observations, classifying samples into five states (initial impulse, decaying impulse, tail, noise, silence). The Viterbi algorithm is used to predict the most likely path through the states. The role of the HMM in this context is to incorporate the time domain into detection, allowing the algorithm to distinguish a target sound from other sounds that share frequency components.

**Anchor 1 (neuroethology_review, Section 6.1):** "Markov models, for example, capture short-timescale dynamics of vocal communication. A typical Markov model of birdsong is simply a transition matrix representing the probability of transitions from each element to each other elements. As Markov models increase in order, they become increasingly capable of capturing long-distance relationships, though high-order Markov models are rarely used in practice because of the number of parameters and amount of data needed to compute them. Approaches such as Hidden Markov Models and Probabilistic Suffix Trees can compute more succinct high-order Markov relationships, though the amount of data needed to capture these deeply contextualized relationships is still a limiting factor in capturing long-range organization with Markov models."

**Anchor 2 (audiomoth_deployment, Section 3.4 / 3.4.4):** "While the bat and cicada detection algorithms rely solely on the frequency domain, gunshots require analysis covering the time domain in order to utilise the signature transition from muzzle blast through different stages of decay. By using the time domain, the algorithm is able to distinguish a gunshot from an insect coincidentally vocalising in the same frequency band. For this reason, the algorithm uses a HMM with three frequency components as observations. The HMM classifies the Goertzel outputs into five possible states: Initial impulse (I), decaying impulse (D), tail (T), noise (N), and silence (S). As the Goertzel-filtered samples are run through the HMM, the Viterbi algorithm is used to predict the most likely path through the states taken by the recording."

---
## Q16 — Category 3 (cross-document synthesis)

**Papers:** neuroethology_review.docx + opensoundscape_methods.docx

**Question:** How does the bioacoustics literature describe the transition from earlier machine-learning approaches (like HMMs and template-matching) to modern deep learning for animal sound classification, and what specific deep learning tools are now available for researchers?

**Expected answer:** The bioacoustics literature describes a clear historical transition. Prior to deep learning, automated birdsong element recognition relied on classical machine-learning algorithms such as Hidden Markov Models, support vector machines, template matching, and k-Nearest-Neighbors labeling, following alongside contemporary speech recognition methods. Current approaches now rely predominantly on recurrent and convolutional neural network architectures, and deep learning — especially with CNNs — is considered the gold standard for most classification tasks. The most powerful and widely adopted underlying machine learning tools are TensorFlow and PyTorch, though they require advanced Python and ML knowledge to use effectively. To make deep learning more accessible to bioacoustics researchers, high-level packages like OpenSoundscape provide a flexible API on top of PyTorch.

**Anchor 1 (neuroethology_review, Section 4):** "Prior to deep learning, automated birdsong element recognition relied on algorithms such as Hidden Markov Models, support vector machines, template matching, or k-Nearest-Neighbors labeling, following alongside contemporary speech recognition algorithms. Like sound event detection, current approaches tend to rely on recurrent and convolutional neural network architectures."

**Anchor 2 (opensoundscape_methods, Section 2):** "Automating the detection of species in audio is a key step in bioacoustic monitoring, and is the focus of several existing software projects. At present, the gold standard method for most classification tasks is considered to be deep learning, especially using convolutional neural networks. The most powerful and widely adopted machine learning tools for deep learning are TensorFlow and PyTorch, both of which require advanced knowledge of Python and machine learning to use effectively. While there are several bioacoustics software tools that aim to provide high-level APIs that simplify the process of training deep learning algorithms for bioacoustic detection, we believe that the flexibility provided by OpenSoundscape will allow the package to be applied to a wider range of bioacoustics problems than is possible with existing software."

---
## Q17 — Category 4 (specific numerical / technical detail)

**Paper:** audiomoth_deployment.docx

**Question:** What is the F1 score, precision, and recall achieved by the AudioMoth bat detection algorithm on the soprano pipistrelle dataset?

**Expected answer:** F1 score of 0.961, precision of 0.994, recall of 0.931.

**Anchor (Section 3.2.5):** "The bat detection algorithm obtained an F1 score of 0.961 from a precision of 0.994 and a recall of 0.931."

---
## Q18 — Category 4 (specific numerical / technical detail)

**Paper:** audiomoth_deployment.docx

**Question:** What is AudioMoth's maximum supported sample rate, and what frequency response range does that give the device?

**Expected answer:** AudioMoth supports sample rates up to 384 kHz, giving the device a frequency response from 20 Hz to 192 kHz.

**Anchor (Section 2.1):** "It captures sound through an analogue MEMS (microelectromechanical) microphone and an analogue preamplifier, supporting sample rates up to 384 kHz to give the device a broad spectrum frequency response from 20 Hz to 192 kHz."

---
## Q19 — Category 4 (specific numerical / technical detail)

**Paper:** neuroethology_review.docx

**Question:** In the neuroethology paper, what specific zebra finch motif characteristics are given as an example of a signal timescale for non-stationary noise reduction?

**Expected answer:** Zebra finch motifs are generally between 0.5 and 1.5 seconds long, repeated one to four times.

**Anchor (Section 2.2):** "For example, zebra finch motifs are generally between 0.5 and 1.5 s long repeated one to four times. Any acoustic event that is outside of that time range could be considered noise."

---
## Q20 — Category 4 (specific numerical / technical detail)

**Paper:** audiomoth_deployment.docx

**Question:** What are the specifications of AudioMoth's processor and memory?

**Expected answer:** AudioMoth uses an Arm Cortex-M4F processor operating at 48 MHz with hardware-enabled floating point arithmetic, has 256 KB of flash memory, and includes an external 256 KB static random access memory (SRAM) chip connected via an external bus interface (EBI).

**Anchor (Section 2.1):** "AudioMoth is designed around an Arm Cortex-M4F processor operating at 48 MHz with hardware-enabled floating point arithmetic and 256 KB of flash memory. It also has an external 256 KB static random access memory (SRAM) chip, which is connected to the micro-controller using an external bus interface (EBI)."

---
## Q21 — Category 5 (out-of-corpus / refusal — Type A: completely off-topic)

**Paper:** N/A (out-of-corpus)

**Question:** What is the average gestation period of an African elephant?

**Expected answer:** A faithful refusal such as: "The provided passages do not contain information about elephant gestation periods. The papers discuss bioacoustic monitoring, OpenSoundscape software, and AudioMoth deployments — none of which cover this topic."

**Note:** Retrieval will return chunks with low semantic similarity to the query (no anchor exists in the corpus). The model passes this question by explicitly refusing to answer, per the prompt instruction "If the passages don't contain enough information, say so explicitly."

---
## Q22 — Category 5 (out-of-corpus / refusal — Type B: adjacent but unanswered)

**Paper:** N/A (Song Meter is mentioned in audiomoth_deployment but the requested specs are not given)

**Question:** What is the sample rate and battery life of the Wildlife Acoustics Song Meter?

**Expected answer:** A faithful refusal such as: "The provided passages mention the Wildlife Acoustics Song Meter series as a commercial acoustic monitoring device costing US $800–1000 per unit, used as a contrast to the open-source AudioMoth. However, the passages do not specify the sample rate or battery life of the Song Meter."

**Note:** This is a Type B refusal — the topic appears in the corpus (high retrieval similarity expected), but the specific requested information is not present. Tests whether the model grounds its answer to what the retrieved chunk *actually says* versus what the chunk is *about*.

---
## Q23 — Category 5 (out-of-corpus / refusal — Type B: adjacent but unanswered)

**Paper:** N/A (TweetyNet is mentioned in neuroethology_review but no accuracy or dataset is given)

**Question:** What accuracy did TweetyNet achieve, and what dataset was it trained on?

**Expected answer:** A faithful refusal such as: "The provided passages mention TweetyNet as an example of a recurrent and convolutional neural network architecture used to capture complex spectro-temporal patterns over long timescales in birdsong. However, the passages do not specify TweetyNet's accuracy or what dataset it was trained on."

**Note:** Type B refusal — TweetyNet is named with a technical-sounding description in the corpus, but performance metrics and training data are not given. Tests whether the model fabricates plausible-sounding numbers in the presence of authoritative-feeling but incomplete context.

---
## Q24 — Category 5 (out-of-corpus / refusal — Type A: completely off-topic)

**Paper:** N/A (underwater/cetacean monitoring with AudioMoth is not covered in the corpus)

**Question:** How does AudioMoth perform in underwater acoustic monitoring of cetacean vocalizations?

**Expected answer:** A faithful refusal such as: "The provided passages describe AudioMoth deployments for terrestrial acoustic monitoring — specifically bat echolocation detection, New Forest cicada detection, and gunshot detection for anti-poaching in rainforests. The passages do not describe underwater deployment or cetacean monitoring."

**Note:** Adversarial Type A — AudioMoth is heavily covered in the corpus, and cetacean acoustic monitoring is a real bioacoustics application, but the corpus contains no information about AudioMoth in that specific context. Tests whether the model resists pattern-matching ("AudioMoth + acoustic + cetacean") into a fabricated answer.

---
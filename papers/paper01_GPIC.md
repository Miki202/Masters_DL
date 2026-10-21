# GPIC: A Giant Permissive Image Corpus for Visual Generation

**Category** - image dataset
**Value** *(why this paper deserves a spotlight as well as a reference to the most valuable section)* - 
It provides a diverce dataset that is also "permissively licensed for both research and commercial use". 
This provides a fix to some known problems with image datasets:
1. Propriety data - a lot of companies do build big datasets, but they use it only internally.
2. Link Breakage - from my understading some datasets rely on URLs, which can break overtime (like LAION <- should look into this more).
3. Copyright issues - self-explanatory.
4. Benchmark saturation - some legacy datasets (like ImageNet) are becoming outdated for evaluating modern generative models.
**What did the authors try to accomplish?** They created an openly accessible dataset that can be used for training visual generative models and also set up a benchmarking protocol and baseline models.
**What were the key elements if the approach?**
- contains 100 million training images (28 trillion pixels);
200 000 validation images and 1 million test ones
- every image is permissively licensed:
    - used only platforms like Flickr & Wikimedia Commons to filter for premissive licenses
    - passed the raw (sommetimes corrupted files, blurry, inappropriate) pool through automated filtering models to discard them
    - deduplication - used SSCD (Self-Supervised Copy Detection)
- images are captioned (alt-text) using Qwen3-VL-4B
- hosted on Hugging Face
**What can you use yourself?** I could use it as a relatively transparent and reproducible resource for training or evaluating text-to-image models.
**What other references do you want to follow?** 
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr,Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh.
Video generation models as world simulators. 2024. URL https://openai.com/research/
video-generation-models-as-world-simulators. 1
- [6] Google. Nano banana 2: Google’s latest ai image generation model. https://blog.google/
innovation-and-ai/technology/ai/nano-banana-2/, February 2026. Accessed: 2026-05-24. 
**What are some limitations?**  I think raining at the full scale still requires substantial compute.
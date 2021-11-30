# FAIG

~~Codes will be released at the end of November, 2021.~~
The codes have been sorted out. Now they are under the internal review process. It may take about two weeks more. Sorry for the inconvenience.

The official PyTorch codes for **NeurIPS 2021 (Spotlight): Finding Discriminative Filters for Specific Degradations in Blind Super-Resolution**.

### :book: Finding Discriminative Filters for Specific Degradations in Blind Super-Resolution (NeurIPS 2021, Spotlight)

> [[arXiv Paper](https://arxiv.org/abs/2108.01070)] &emsp; [Project Page] &emsp; [YouTube Video] &emsp; [B站] &emsp; [Poster] &emsp; [PPT slides]<br>
> [Liangbin Xie](https://www.researchgate.net/profile/Liangbin-Xie), [Xintao Wang](https://xinntao.github.io/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Zhongang Qi](https://scholar.google.com/citations?user=zJvrrusAAAAJ&hl=en), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Tencent ARC Lab; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

<p align="center">
  <img src="assets/FAIG_teaser.jpg">
</p>
For a blurry input (①) and noisy input (⑤), the one-branch SRResNet for blind SR could remove blur (②) and noise (⑥), respectively. When we mask the 1% deblurring filters (discovered by the proposed FAIG), the corresponding network function of deblurring is eliminated (③) while the function of denoising is maintained (⑦). Similar phenomenon happens (④ and ⑧) when we mask the 1% denoising filters in the same network.

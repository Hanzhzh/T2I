# 该项目代码有所改变，我将尽快修改代码，并给出readme文件

## 项目适用的场景：


## 项目背景
### 背景1：

由于自己从入学以来（一年半）都在参与某一个卫星/无人机/地面装备协同作战的军工项目，项目中的其中一个研究课题就是跨视角的图像匹配与检索任务（检索数据集中同一目标的不同视角图像），数据集的样例如下（非真实数据）：
<div style="display: flex; justify-content: space-between; align-items: center;gap:10%;">
  <img src="https://github.com/user-attachments/assets/60b928b5-59ad-488c-b147-b943a2a03d01" style="width:20%; height:auto;">
  <img src="https://github.com/user-attachments/assets/d41359b5-fd7c-4df6-bda6-4d3c75362673" style="width:20%; height:auto;">
  <img src="https://github.com/user-attachments/assets/a69e0877-215f-4a8f-8c30-6c5514c29be6" style="width:20%; height:auto;">
</div>

第一想法是在无人机机载或卫星机载上直接进行图像匹配与检索，但实际上卫星与无人机的计算资源与存储资源十分有限，随着运行时间推进，无法满足图像数据库的存储需求，而且由于设备的移动速度很快，对匹配与检索速度的要求非常高，图像与图像进行检索显然过于耗时了。

### 背景2：

近年来，图像描述生成(Image Caption)领域不断发展，其主要目标是根据输入的图像信息，生成相应的文字描述，从而完成对图像内容的准确描述。为了实现图像描述的任务，常见的方法是采用编码器-解码器（encoder-decoder）的结构。这种结构可以将输入的图像信息通过编码器进行抽象和提取，得到一个表示图像特征的向量。然后，解码器将这个向量作为输入，逐步生成与图像内容相对应的文字描述。这种结构的实现中，常常使用transformer作为主体机构。

目前所发表的论文中大都是通过构建或已有的大规模的图片-空间描述数据集并结合预训练的视觉语言模型来完成这一目标，但不知是数据集中的描述文本普遍过短还是视觉语言模型的能力不足，导致在我所实验数据集上的效果不佳，也可以说是几乎没有效果。我思考的原因可能是多数论文中的模型更倾向于针对某一个数据集，针对该数据集进行了精心的调优，使其在面对其他更多领域的数据集时难以给出满意的表现（当然，复现论文也比较消耗精力，也有可能是我所看的这些方法的不够前沿，但无论如何，我想使用一种更为简单有效的方法来完成这一事情）。

### 总结：
  结合以上的背景与思路，考虑使用多模态的方法来解决，即对所有的图像进行文本描述生成，在云端使用文本与图像相互检索的方式加快计算速度。
  
  在这个思路下，文本的生成质量起到了关键性的作用，于是对一些性能较好的描述生成模型进行实验，发现这些模型在这一类数据集上生成的文本不仅很短而且质量不好，根本无法支撑通过文本区分图像的目的。
  
  既然普通模型无法满足需求，那么考虑利用大模型的优势来解决这一问题会是什么效果呢？答案是：效果很好。
  
  对于一张图像，可以通过优化的Prompt+大模型得到足够长足够细节的描述文本，在实验数据的跨视角检索和匹配中，其图像到文本和文本到图像的Recall@1分别达到76%与72%.

## 项目方案
拟采用的方案是：
- 精心设计Prompt，不断根据生成文本的效果进行优化，最终得到了多个针对不同视角不同细节的Prompt，细节程度可以支撑起整体逻辑。
- 将拍摄的历史图像数据集进行离线文本描述生成，即统一在地面进行处理，这样部分解决了载荷在实时计算时的计算资源有限问题；
- 载荷上使用文本替代图像，在进行检索时，使用文本与图像的跨模态检索，避免了直接进行图像检索匹配，减小了计算压力。
- 针对视频处理，可以采用关键帧的方法，选取关键帧作为输入，进而得到视频的描述。

这个方案通过将检索匹配与大语言模型结合，探索了一种新的解决方式。



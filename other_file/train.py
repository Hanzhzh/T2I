from torch.cuda.amp import autocast
from other_file.config import Config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#加权损失函数
from loss.NCELoss import weighted_infonce_loss


# 对图像做预处理: processor会将PIL的images转换成模型所需要的tensor格式，还会进行归一化、resize等等
# 定义训练函数
def train_one_epoch(model, dataloader, optimizer, device, scaler, epoch, config: Config, writer: SummaryWriter, current_alpha: float, current_belta: float):
    model.train()
    total_loss = 0

    # 动态调整 alpha
    alpha_increment = (config.ALPHA_FINAL - config.ALPHA_INITIAL) / config.EPOCHS
    alpha = current_alpha + alpha_increment
    alpha = min(alpha, config.ALPHA_FINAL)  # 确保 alpha 不超过 ALPHA_FINAL
    print(f"Epoch {epoch+1}: 使用的 alpha = {alpha:.4f}")

    # 动态调整 belta
    belta_increment = (config.belta_final - config.belta_initial) / config.EPOCHS
    belta = current_belta + belta_increment
    belta = min(belta, config.belta_final)  # 确保 alpha 不超过 ALPHA_FINAL
    print(f"Epoch {epoch + 1}: 使用的 belta = {belta:.4f}")


    for batch_idx, (images, text_embeddings, image_paths, subset_ids, sample_indices) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device).float()

        batch_size = images.size(0)  # 获取当前批次的大小
        if batch_size == 0:
            print(f"空批次，跳过 batch_idx={batch_idx}")
            continue  # 跳过空批次

        text_embeddings = text_embeddings.to(device).float()

        optimizer.zero_grad()

        with autocast():

            # #已经标准化，数据变为零均值和单位标准差，说明已经标准化。
            # inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            # # 前向传播：Dinov2Model在transformers中的forward方法接受的参数与CLIP不同，需要传入**inputs
            # # **inputs会包含pixel_values等张量，并由DINOv2模型处理
            # outputs = model(**inputs)
            # # outputs是Dinov2Model的输出，一般包含last_hidden_state和pooler_output
            # # pooler_output是一个[batch_size, hidden_size]的张量，可以作为图像的全局表示特征使用
            # image_features = outputs.pooler_output.float()
            # # text_features即之前加载的npy嵌入，不需要投影了(取决于你的下游需求)
            # text_features = text_embeddings

            # 前向传播
            outputs = model(images, text_embeddings)
            # outputs通常返回(image_features, text_features)
            image_features, text_features = outputs[:2]

            # 使用你的加权infonce损失函数
            loss = weighted_infonce_loss(image_features, text_features, subset_ids, alpha=alpha, belta=belta)

        # 反向传播与优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} 训练损失：{avg_loss:.4f}")

    # 记录到 TensorBoard
    writer.add_scalar('Loss/Train', avg_loss, epoch+1)
    writer.add_scalar('Hyperparameters/Alpha', alpha, epoch+1)

    return avg_loss, alpha, belta
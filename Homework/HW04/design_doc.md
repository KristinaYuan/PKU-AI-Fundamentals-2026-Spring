姓名：袁佳逸

学号：2400010962

# Transformer 作业实验报告

## 理解数据流

请根据 `nmt_de2en.py` 的实现，完成以下任务：

batch size 记为 `B`，源语言序列长度记为 `S`，目标语言输入序列长度记为 `T`，词嵌入维度记为 `E`，目标语言词表大小记为 `V_tgt`。

### 主要 Tensor 的 shape

请补全下表：

（提示：可以在代码中直接打印 shape 进行观察）

| Tensor             | shape           | 含义或作用                                                         |
| ------------------ | --------------- | ------------------------------------------------------------------ |
| `src`              | `[S, B]`        | 源语言 token id 序列                                               |
| `tgt`              | `[T + 1, B]`    | 目标语言 token id 序列                                             |
| `tgt_input`        | `[T, B]`        | 目标语言输入 token id 序列                                         |
| `src_emb`          | `[S, B, E]`     | src 的词嵌入结果                                                   |
| `tgt_emb`          | `[T, B, E]`     | tgt_input 的词嵌入结果                                             |
| `src_mask`         | `[S, S]`        | 用于控制源语言 token 之间哪些位置可以互相注意，本实验中为全 0 矩阵 |
| `tgt_mask`         | `[T, T]`        | 用于控制目标语言 token 之间哪些位置可以互相注意                    |
| `src_padding_mask` | `[B, S]`        | 源语言序列的填充掩码                                               |
| `tgt_padding_mask` | `[B, T]`        | 目标语言序列的填充掩码                                             |
| `outs`             | `[T, B, V_tgt]` | 模型输出的 logits                                                  |
| `logits`           | `[T, B, V_tgt]` | 模型输出的 logits                                                  |
| `tgt_out`          | `[T, B]`        | 目标语言输出 token id 序列                                         |

请回答：

Q1：在验证集上测试模型和调用 translate 函数都会进行翻译，请指出它们运行逻辑上的主要不同点。

A1：
- 验证集测试：使用 `tgt` 作为输入 Decoder，一次性并行得到所有位置的 `logits`，再与 `tgt_out` 计算 `loss` 和准确率，评估模型性能
- 调用 translate 函数：从 `BOS` 开始每次只生成一个 token，再把生成的 token 接回去继续生成，直到遇到 `EOS` 或达到最大长度，逐步生成目标

### Encoder & Decoder

请根据 `transformer.py` 中的定义，判断 Encoder 和 Decoder 分别用到了哪些信息，并补全下表。

| 是否用到了以下信息 | Encoder | Decoder |
| ------------------ | ------- | ------- |
| `src_emb`          | Y       | N       |
| `tgt_emb`          | N       | Y       |
| `src_mask`         | Y       | N       |
| `tgt_mask`         | N       | Y       |
| `memory`           | N       | Y       |

请回答：

Q2：memory 的 shape 是多少？（使用 `B, S, T, E` 等记号表示）

A2: `[S, B, E]`

Q3: Encoder 和 Decoder 分别有几层？

A3: Encoder 有 3 层，Decoder 有 3 层。

Q4: 请把单层 Encoder 和单层 Decoder 的前向传播过程的代码粘贴到下方。

A4:
```
# EncoderLayer forward
def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
```
```
# DecoderLayer forward
def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
```

## Embedding 实验

### 删除 EOS 的影响

本部分建议在 `nmt_de2en.py` 上完成。德英数据集较小，训练更快，适合做对比实验。

修改 `tensor_transform`，删除句尾的 `EOS_IDX`，重新训练模型，并观察翻译结果，填写下表：

| 设置     | 最终训练 loss | 最终验证 loss | 原句                                           | 翻译结果                                                       | 现象     |
| -------- | ------------- | ------------- | ---------------------------------------------- | -------------------------------------------------------------- | -------- |
| 保留 EOS | 0.783         | 1.941         | Eine Gruppe von Menschen steht vor einem Iglu. | A group of people stand in front of an igloo.                  | 正常     |
| 删除 EOS | 0.829         | 2.064         | Eine Gruppe von Menschen steht vor einem Iglu. | A group of people stand in front of an igloo . UTEP . birdie . | 异常结尾 |

### 词嵌入观察

训练 `nmt_zh2en.py` 模型，并参考 `docs/cli_tools.md` 中的命令行说明，查询若干词在 embedding 空间中的最近邻。

请选择至少 5 个英文或中文词进行观察。

| 查询词 | 最相近 token 1 | 最相近 token 2 | 最相近 token 3 |
| ------ | -------------- | -------------- | -------------- |
| apple  | Pandemic       | Agreed         | Animal         |
| 苹果   | 安倍           | 东京           | 马纳福特       |
| spring | night          | week           | March          |
| 语言   | 种族           | 文字           | 言语           |
| 自然   | 物竞天择       | 本性           | 天生           |

## 温度实验

本部分在 `nmt_zh2en.py` 上完成。

温度会影响生成结果，温度越高越自由，翻译模型一般会让温度等于 0（greedy decode）。请选择至少 3 个中文句子，每个句子分别测试 4 种温度，并记录结果。

`docs/cli_tools.md` 中记录了交互式翻译和设置温度的方法。修改温度不需要重新训练模型，完成一次训练后会直接加载已有的 checkpoint。

| 中文输入                                                                                             | temperature | 英文输出                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 但目前这方面的投入还远远不够。                                                                       | 0           | But this is far from enough .                                                                                                                                                                        |
| 但目前这方面的投入还远远不够。                                                                       | 0.5         | But the investments are far from enough .                                                                                                                                                            |
| 但目前这方面的投入还远远不够。                                                                       | 1           | But this is far from enough .                                                                                                                                                                        |
| 但目前这方面的投入还远远不够。                                                                       | 1.5         | But this demand is much enough mutual .                                                                                                                                                              |
| 新技术支持的金融工具为某些人赚取巨额利润创造了机遇。                                                 | 0           | Financial tools that support new technology create opportunities for some to receive large profits .                                                                                                 |
| 新技术支持的金融工具为某些人赚取巨额利润创造了机遇。                                                 | 0.5         | Financial tools that are backed by new technology create opportunities for some to earn large profits .                                                                                              |
| 新技术支持的金融工具为某些人赚取巨额利润创造了机遇。                                                 | 1           | Financial tools pro - technology support create opportunities for some happen shrinking profits .                                                                                                    |
| 新技术支持的金融工具为某些人赚取巨额利润创造了机遇。                                                 | 1.5         | recovered sent opportunities equally rapidly best - reap least starting shrinking financial contain significant way enabled everyone to package grew profit game                                     |
| 反恐国际合作显著增强，部分是因为，各国政府都觉得需要在这一领域加强合作，尽管它们在其他领域颇有分歧。 | 0           | The international cooperation on terrorism has increased significantly , partly because governments feel that need to strengthen cooperation in this area , though they are divided in other areas . |
| 反恐国际合作显著增强，部分是因为，各国政府都觉得需要在这一领域加强合作，尽管它们在其他领域颇有分歧。 | 0.5         | International cooperation on terrorism has increased significantly , partly because governments feel that need to strengthen cooperation in this area , though they are divided on other areas .     |
| 反恐国际合作显著增强，部分是因为，各国政府都觉得需要在这一领域加强合作，尽管它们在其他领域颇有分歧。 | 1           | International cooperation on terrorism universal , partly because their governments shortage to strengthen cooperation in this area , though they are and divided over other areas .                 |
| 反恐国际合作显著增强，部分是因为，各国政府都觉得需要在这一领域加强合作，尽管它们在其他领域颇有分歧。 | 1.5         | Treasury whole shortage balance reached grow is tight , partly with fear about and more cooperation decade in resistance , but in no way they exports more .                                         |

## 超参数修改

本部分在 `nmt_de2en.py` 上完成。请使用相同数据集和相同训练轮数，对比不同模型规模对训练速度、loss、显存占用和过拟合情况的影响。

请分别训练以下三组超参数。

| 配置 | `EMB_SIZE` | `NHEAD` | `FFN_HID_DIM` | Encoder layers | Decoder layers |
| ---- | ---------- | ------- | ------------- | -------------- | -------------- |
| Tiny | 128        | 4       | 256           | 2              | 2              |
| Base | 512        | 8       | 512           | 3              | 3              |
| Wide | 512        | 8       | 1024          | 4              | 4              |

训练脚本会在 `log/de2en/` 下导出 CSV，请根据 CSV 填写下表。

| 配置 | 最终 train loss    | 最终 val loss      | 单 epoch 平均时间 | 最大显存占用    |
| ---- | ------------------ | ------------------ | ----------------- | --------------- |
| Tiny | 2.0085060055560477 | 2.2060405239462852 | 12.13s            | 652.3896484375  |
| Base | 0.7826906429513436 | 1.9413174018263817 | 39.08s            | 1465.8642578125 |
| Wide | 0.8383088318655669 | 1.8997707590460777 | 52.03s            | 1881.650390625  |

Q5: 请根据实验结果，简述不同模型规模产生的影响。

A5:
- 模型越大，训练时间和显存占用越高，说明更大的模型会明显带来更高的计算和存储成本。
- 较大的模型通常具有更强的拟合能力，相比于 Tiny，Base 和 Wide 的验证集表现明显更好，说明增大模型规模确实有助于提升翻译效果。
- Base 和 Wide 在后期都出现了轻微过拟合现象：训练损失继续下降，但验证损失在最优点之后不再明显改善，甚至略有波动。相比之下，Tiny 虽然不太容易过拟合，但主要问题是欠拟合。
- Tiny 训练最快、资源消耗最少，但性能不足；Wide 性能最好但代价最高，而且相对 Base 的提升有限；Base 在性能和训练成本之间取得了较好的平衡，因此是这组实验中性价比最高的配置。

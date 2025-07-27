Looking at your current setup with CLIP ViT encoder + GPT-2 decoder still yielding BLEU ~0.1, there are several fundamental issues that need to be addressed. Let me suggest specific approaches to improve the quality:

## **1. Major Architectural Issues with Current GPT-2 Decoder**

### **Problems I see in your current implementation:**
- **Vocabulary mapping complexity**: The custom-to-GPT2 vocabulary mapping is lossy and error-prone
- **No image conditioning**: GPT-2 doesn't see image features during generation
- **Suboptimal fusion**: Image features aren't properly integrated into the text generation process
- **Training mismatch**: GPT-2 wasn't trained for image captioning

### **Better Decoder Architectures:**

#### **A. Image-Conditioned Transformer Decoder (Recommended)**
```python
# Replace GPT-2 with a custom transformer that properly conditions on image features
class ImageConditionedTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=6):
        # Custom transformer with cross-attention to image features
        # Image features fed as "encoder" output to transformer decoder
```

#### **B. Vision-Language Models (Best Option)**
- **Use BLIP/BLIP-2**: Specifically designed for image captioning
- **Use GIT (GenerativeImage2Text)**: Microsoft's image captioning model
- **Use OFA (One For All)**: Unified vision-language model

## **2. Immediate Improvements for Current Architecture**

### **Fix GPT-2 Integration:**
- **Remove vocabulary mapping**: Use GPT-2's native vocabulary entirely
- **Proper image injection**: Prepend image features as "image tokens" to text sequence
- **Cross-modal attention**: Add cross-attention layers between image and text

### **Better Training Strategy:**
- **Two-stage training**: 
  1. Freeze GPT-2, train only image projection layer
  2. Fine-tune entire model with very low learning rate
- **Prompt-based approach**: Train GPT-2 to respond to "Describe this image:" prompts

## **3. Data and Training Improvements**

### **Dataset Enhancements:**
- **Use COCO Captions**: Much larger and higher quality than Flickr
- **Combine datasets**: Flickr8k + Flickr30k + COCO + Conceptual Captions
- **Data quality filtering**: Remove poor quality or repetitive captions

### **Advanced Training Techniques:**
- **SCST (Self-Critical Sequence Training)**: Optimize directly for BLEU/CIDEr
- **Reinforcement Learning**: Use BLEU/ROUGE as reward signals
- **Curriculum Learning**: Start with shorter, simpler captions

## **4. CLIP-Specific Optimizations**

### **Better CLIP Usage:**
- **Use CLIP text encoder too**: For better text-image alignment
- **CLIP-guided training**: Use CLIP similarity as additional loss
- **Multiple CLIP features**: Extract features from different layers

### **CLIP Models to Try:**
- **ViT-L/14**: Larger model for better features (if computationally feasible)
- **ViT-B/16**: Better spatial resolution than ViT-B/32

## **5. Loss Function and Optimization**

### **Multi-objective Training:**
```python
# Combine multiple losses
total_loss = (
    cross_entropy_loss + 
    0.1 * clip_similarity_loss + 
    0.05 * diversity_loss
)
```

### **Advanced Loss Functions:**
- **Focal Loss**: Handle vocabulary imbalance
- **Label Smoothing**: Reduce overconfidence
- **CLIP Contrastive Loss**: Ensure image-text alignment

## **6. Specific Code Improvements**

### **For your current GPT-2 decoder:**

#### **A. Fix Image Feature Integration:**
```python
# Instead of complex vocab mapping, use image as context
def forward(self, image_features, captions):
    # Project image features to GPT-2 embedding space
    image_embeds = self.image_projection(image_features)  # [batch, embed_dim]
    
    # Concatenate image embedding with text embeddings
    text_embeds = self.gpt2_model.transformer.wte(captions)
    
    # Prepend image as first token
    combined_embeds = torch.cat([image_embeds.unsqueeze(1), text_embeds], dim=1)
    
    # Pass through GPT-2 with combined embeddings
    outputs = self.gpt2_model.transformer(inputs_embeds=combined_embeds)
```

#### **B. Better Generation Strategy:**
```python
def generate_caption(self, image_features, max_len=20):
    # Start with image features as context
    image_embeds = self.image_projection(image_features)
    
    # Use beam search instead of greedy decoding
    # Or use nucleus sampling (top-p)
    
    # Ensure image context is maintained throughout generation
```

## **7. Architectural Alternatives (Recommended)**

### **A. Switch to Proven Architecture:**
```python
# Use established image captioning architecture
class ShowAttendTellModel(nn.Module):
    # CLIP encoder + Transformer decoder with attention
    # Proven to work well for image captioning
```

### **B. Use Pre-trained Vision-Language Models:**
- **Fine-tune BLIP**: Pre-trained on image-text pairs
- **Fine-tune GIT**: State-of-the-art image captioning
- **Use InstructBLIP**: For instruction-following image description

## **8. Hyperparameter and Training Optimization**

### **Learning Rate Strategy:**
- **Different LR for different components**: Lower LR for pre-trained parts
- **Warmup and decay**: Proper learning rate scheduling
- **Gradient clipping**: Prevent exploding gradients

### **Batch Size and Memory:**
- **Larger batch sizes**: 256-512 if possible
- **Gradient accumulation**: Simulate larger batches
- **Mixed precision**: Use FP16 for faster training

## **9. Evaluation and Debugging**

### **Better Metrics:**
- **CIDEr and SPICE**: More meaningful than BLEU for captioning
- **CLIPScore**: Semantic similarity between image and caption
- **Human evaluation**: Sample-based quality assessment

### **Debugging Steps:**
- **Check vocabulary coverage**: Ensure important words are included
- **Analyze attention patterns**: Verify model looks at relevant image regions
- **Caption length analysis**: Check if captions are too short/long
- **Diversity metrics**: Ensure captions aren't too repetitive

## **Priority Recommendations:**

### **Immediate (Quick Wins):**
1. **Fix image feature injection** in GPT-2 decoder
2. **Use larger datasets** (add COCO)
3. **Implement beam search** for generation
4. **Add CLIP similarity loss**

### **Medium-term (Better Results):**
1. **Replace GPT-2** with custom transformer decoder
2. **Implement SCST training** for direct metric optimization
3. **Use ViT-L/14** CLIP model
4. **Add reinforcement learning**

### **Long-term (Best Results):**
1. **Switch to BLIP/GIT** architecture
2. **Multi-modal pre-training** approach
3. **Ensemble methods**
4. **Advanced attention mechanisms**

The key insight is that **GPT-2 wasn't designed for image captioning**, so the current approach is fundamentally limited. Moving to architectures specifically designed for vision-language tasks will yield much better results.
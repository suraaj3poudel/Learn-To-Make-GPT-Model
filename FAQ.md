# 📚 Frequently Asked Questions - Phase 1

This document contains real questions from learners going through the ML journey. Use this to clarify concepts!

---

## 🧠 Neural Networks Basics

### **Q: Do I need to pass parameter names like `temp=30` or just `30`?**
**A:** Both work! `simple_neuron(30, 1, 5)` is fine. Named arguments like `temperature=30` are just for clarity in examples.

---

### **Q: Do weights need to add up to 1?**
**A:** No! Weights can be ANY numbers - positive, negative, large, small. The sum doesn't matter. In the examples they sum to 1.0 just for simplicity, but during training they'll be whatever minimizes error.

---

### **Q: How is "if score > 0" defined as the threshold?**
**A:** That's the activation function! The code `decision = "GO" if score > 0 else "STAY"` defines it. The threshold of 0 is arbitrary - bias actually controls the effective threshold.

---

### **Q: If we increase weights, won't the score easily surpass 0?**
**A:** Yes! The bias is what really controls the threshold. Negative bias = harder to activate, positive bias = easier to activate. The "0" threshold is just convention - bias adjusts the actual decision boundary.

---

## 🎨 Network Architecture

### **Q: Why multiple boxes in the hidden layer? Isn't it just one layer?**
**A:** Each box is a separate neuron with its own weights and bias! Multiple neurons in one layer learn different patterns:
- Neuron 1 might detect edges
- Neuron 2 might detect textures
- Neuron 3 might detect shapes

Together they learn complex features. One neuron = one simple pattern. Many neurons = rich understanding!

---

### **Q: Why multiple output neurons? Don't we want just one result?**
**A:** Depends on the problem!
- **1 output:** Yes/No questions, single predictions (house price)
- **Multiple outputs:** Multi-class classification (cat/dog/bird), multi-label (beach, sunset, people)
- **GPT example:** 50,000 outputs (one for each possible next word!)

---

## 🎓 Learning & Training

### **Q: How did we come up with `weight += learning_rate * error * x`?**
**A:** It comes from calculus (gradient descent)!

The formula is derived by taking the derivative of the error with respect to the weight:
```
Error = (target - prediction)²
prediction = weight × x

∂Error/∂weight = -2 × error × x
# The "2" is absorbed into learning_rate, giving us:
weight += learning_rate × error × x
```

**Intuition:** The formula tells us which direction and how much to adjust the weight to reduce error.

---

### **Q: What does `∂Error/∂weight = -error × x` actually mean?**
**A:** It's the gradient (slope) that shows:
- **Direction:** Should we increase or decrease the weight?
- **Magnitude:** How much should we change it?

**Breakdown:**
- `error` = how wrong we are (target - prediction)
- `x` = input value (scales the adjustment)
- If error > 0 and x > 0 → increase weight
- If error < 0 and x > 0 → decrease weight
- Large x → bigger adjustments (input had more influence)

---

### **Q: If x is huge (like 1000) vs small (like 1), won't weights update differently?**
**A:** YES! This is a real problem! 

```python
x = 1000 → weight += 0.01 * 1 * 1000 = +10  (HUGE!)
x = 1    → weight += 0.01 * 1 * 1 = +0.01   (tiny)
```

**Solution:** Normalize/scale inputs before training:
```python
x_scaled = (x - min) / (max - min)  # Scale to 0-1 range
```

This keeps all inputs in similar ranges and prevents unstable training. You'll learn this in detail in upcoming notebooks!

---

### **Q: How do we calculate learning rate? Is there a gold standard after normalizing data?**
**A:** We decide it! No perfect formula, but common guidelines:

**Starting values:**
- `0.001` - Most common, safe default
- `0.01` - Also popular
- `0.0001` - For careful/stable training

**The tradeoff:**
- Too large → unstable, overshoots
- Too small → very slow training
- Just right → smooth convergence

**How to find it:**
1. Start with 0.001 or 0.01
2. Watch the loss during training:
   - Loss decreases smoothly? ✅ Good!
   - Loss explodes? → Too high, reduce by 10x
   - Loss barely moves? → Too low, increase by 10x
3. Experiment and adjust

**Advanced:** Learning rate schedules (start high, decrease) or adaptive optimizers (Adam, RMSprop) automatically adjust it.

---

## 🎨 Activation Functions

### **Q: When do I use Sigmoid vs ReLU vs Tanh? Why these ranges?**
**A:** Each has specific use cases:

**Sigmoid (0 to 1):**
- **Use:** Output layer for probabilities/binary classification
- **Examples:** 
  - Is this spam? → 0.85 (85% spam)
  - Will customer buy? → 0.23 (23% chance)
- **Why 0-1?** Matches probability range (0% to 100%)!

**ReLU (keeps positive, zeros negative):**
```python
ReLU(5) = 5, ReLU(-3) = 0
```
- **Use:** Hidden layers (most popular!)
- **Why it helps:**
  - Fast: just `max(0, x)`
  - Prevents vanishing gradients
  - Sparse activation (only some neurons fire - brain-like!)
- **Example:** Image recognition - some features activate (edge detected), others don't

**Tanh (-1 to 1):**
- **Use:** When outputs are naturally centered around zero
- **Examples:**
  - Sentiment: -1 (negative), 0 (neutral), +1 (positive)
  - Stock: -1 (down), 0 (flat), +1 (up)
- **Why -1 to +1?** Better for bipolar values!

**Common pattern:**
```
Input → [ReLU layers] → Sigmoid output
        ↑ detect features  ↑ final probability
```

---

## 💬 General Learning Questions

### **Q: How are tokens calculated?**
**A:** Tokens = your question + my answer + context (files open, etc.)
- Roughly 1 token ≈ 4 characters
- Short question: ~5-10 tokens
- Detailed answer: ~500-1000 tokens
- You have 1M tokens = room for 1000+ exchanges!

---

## 📝 More Questions Coming...

This FAQ will be updated as you progress through the notebooks!

---

*Last updated: Phase 1, Lesson 1 - Learning Section*

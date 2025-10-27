# --------------------------------------------------------------
# Transformer Bloğu 
# --------------------------------------------------------------
# Girdi → LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kısa yol bağlantısı
#       → LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kısa yol bağlantısı
# --------------------------------------------------------------
# Bu dosya, özellikle (+) Kısa yol bağlantısı (skip connections) kavramını örnekle açıklar.
# Transformer mimarisinde skip connection, gradyan kaybolmasını engeller
# ve bilgi akışını katmanlar arasında korur.
# Bu yapı ilk olarak “ResNet” mimarisinde tanıtılmıştır ve
# derin modellerde eğitimi kararlı hale getirmek için kullanılır.
# --------------------------------------------------------------

import torch
import torch.nn as nn

# GELU aktivasyon fonksiyonu tanımı
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        )) 

# Derin bir sinir ağı örneği: 5 gizli katman + isteğe bağlı kısa yol bağlantısı
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        # Katmanlar sıralı şekilde tanımlanıyor
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # Eğer giriş ve çıkış boyutları eşitse kısa yol bağlantısı eklenir
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# Modeldeki ağırlıkların gradyan büyüklüklerini ekrana yazdırma fonksiyonu
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# --------------------------------------------------------------
# Örnek kullanım: Skip connection etkisinin gözlemlenmesi
# --------------------------------------------------------------
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., 1.]])

# Kısa yol bağlantısı olmadan model
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)
# output:
# layers.0.0.weight has gradient mean of 0.00015044710016809404
# layers.1.0.weight has gradient mean of 0.00013967031554784626
# layers.2.0.weight has gradient mean of 0.0006069596274755895
# layers.3.0.weight has gradient mean of 0.0011254228884354234
# layers.4.0.weight has gradient mean of 0.004502863623201847

# Kısa yol bağlantısı olan model
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
# output:
# layers.0.0.weight has gradient mean of 0.231016606092453
# layers.1.0.weight has gradient mean of 0.2370775043964386
# layers.2.0.weight has gradient mean of 0.34811022877693176
# layers.3.0.weight has gradient mean of 0.13332870602607727
# layers.4.0.weight has gradient mean of 1.8219515085220337

# --------------------------------------------------------------
# Açıklama:
# - Skip connection, derin ağlarda gradyanların kaybolmasını önler.
# - Yukarıdaki örnekte, kısa yol eklenince gradyan büyüklükleri ciddi şekilde artmıştır.
# - Bu, modelin daha iyi öğrenmesini sağlar.
# --------------------------------------------------------------

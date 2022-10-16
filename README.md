# mini_auto_grad

## Derivative rules

| Name                         | Function                                      |
| ---------------------------- | --------------------------------------------- |
| Definition                   | $\frac{dx}{dx} = 1$                           |
| Constant rule                | $\frac{d}{dx}[c] = 0$                         |
| Constant multiplication rule | $\frac{d}{dx}[c f(x)] = cf'(x)$               |
| Power rule                   | $\frac{d}{dx}[x^n] = n x^{(n-1)}$             |
| Sum rule                     | $\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$   |
| Difference rule              | $\frac{d}{dx}[f(x) - g(x)] = f'(x) - g'(x)$   |
| Chain rule                   | $\frac{dy}{dx} = \frac{dy}{dz} \frac{dz}{dx}$ | 

# MP Doom — Doom (1993) com MediaPipe para controles imersivos

Mapeia gestos de mão e pose (MediaPipe) para controles do Doom clássico com a biblioteca ViZDoom e WAD Fredoom. Permite controlar movimento, mira e disparo usando gestos.

## Grupo TB — Checkpoint 2 — Cognitive Computing & Computer Vision

Ana, Felipe, Isabella, Lucas e Martin

## Pré-requisitos

- OS Windows
- Python 3.11+
- pip

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/classarnaldojr/cp2-4sir-tb.git
```

2. Crie e ative o ambiente virtual:

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Windows cmd
.\venv\Scripts\Activate.bat
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como jogar

Inicie o jogo:

```bash
python mpdoom.py
```

- Pressione `SPACE` para exibir tracking
- Pressione `ESC` para sair

## Comandos de gesto

Gestos são imersivos e combinados com fluidez!

| Gesto | Ação no jogo | Definição (resumida, baseada no código) |
|---|---:|---|
| Antebraço direito estendido | Andar pra frente | Index TIP distante (vertical) do ombro |
| Braço direito retraído | Parar | Index TIP próximo (vertical) do ombro |
| Antebraço direito lateral (movimento) | Mirar / mover câmera | Pulso movimentando (horizontal) |
| Dedo indicador: estende → retraído | Atirar | Comparação de posições de 4 pontos do Index (TIP/DIP/PIP/MCP) |
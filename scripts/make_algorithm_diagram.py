#!/usr/bin/env python3
"""
Generate the protocol sequence diagram for SecureDrive-FL with our Compute-Budgeted
and Energy-Adaptive Extensions, styled identically to algorithm_reference.png.

Original paper idea in BLACK.
Our modifications in RED (underlined).
"""

import os
import subprocess
import sys
from PIL import Image

def generate_tikz_source():
    tex = r'''\documentclass{article}
\usepackage[margin=0.35in,paperwidth=16.5in,paperheight=23.5in]{geometry}
\usepackage{tikz}
\usetikzlibrary{decorations.pathreplacing,calc,arrows.meta}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\pagestyle{empty}

% Define custom colors matching publication standard
\definecolor{algored}{RGB}{204, 0, 0}
\newcommand{\myred}[1]{\underline{\textcolor{algored}{#1}}}
\newcommand{\redtext}[1]{\textcolor{algored}{#1}}
\newcommand{\redbold}[1]{\underline{\textcolor{algored}{\textbf{#1}}}}

\begin{document}
\begin{center}
\begin{tikzpicture}[>=Stealth, font=\small]

% Coordinates
\def\xbrace{-1.8}
\def\xclient{0.0}
\def\xserver{14.5}

% -------------------------------------------------------------
% TOP ACTOR BOXES
% -------------------------------------------------------------
% Stacked client cards (Vehicles / Clients) centered at \xclient
\draw[fill=white, thick] (\xclient-2.7, 0.28) rectangle ++(5.6, 0.68);
\draw[fill=white, thick] (\xclient-2.8, 0.14) rectangle ++(5.6, 0.68);
\node (clientbox) [draw, fill=white, thick, minimum width=5.6cm, minimum height=0.68cm] 
    at (\xclient, 0.0) {\textbf{\large Vehicles / Clients $i \in \mathcal{C}_n$}};

% Single Server box centered at \xserver
\node (serverbox) [draw, fill=white, thick, minimum width=5.6cm, minimum height=0.68cm] 
    at (\xserver, 0.0) {\textbf{\large Aggregation Server}};

% Lifeline bottom coordinate
\def\ybottom{-22.8}
\draw[dashed, thick] (clientbox.south) -- (\xclient, \ybottom);
\draw[dashed, thick] (serverbox.south) -- (\xserver, \ybottom);

% =============================================================
% ROUND 0: Model Broadcast & Dynamic Capacity Profiling
% =============================================================
\def\yRzeroTop{-0.3}
\def\yRzeroBot{-3.5}

\draw[decorate, decoration={brace, amplitude=6pt, mirror}, thick] 
    (\xbrace, \yRzeroBot) -- (\xbrace, \yRzeroTop)
    node[midway, left=9pt, align=right] {
        \textbf{\large Round 0:}\\[3pt]
        \textbf{Model Broadcast}\\[2pt]
        \& \myred{\textbf{Budget Profiling}}
    };

% Server local: Init model
\node[anchor=east, align=right] at (\xserver-0.3, -0.60) {
    Initialize global model $W^{(0)}$
};

% Arrow Server -> Client
\draw[->, thick] (\xserver, -1.35) -- (\xclient, -1.35)
    node[midway, above=3pt] {\textbf{Broadcast global model $W^{(n)}$ to participating clients $\mathcal{C}_n$}};

% Client local actions
\node[anchor=west, align=left] at (\xclient+0.3, -1.90) {
    Receive global model $W^{(n)}$; initialize local training state
};
\node[anchor=west, align=left] at (\xclient+0.3, -2.45) {
    \myred{Profile vehicular hardware capacity tier $c_i \in [0.1, 1.0]$ and dynamic availability $a_{i,t} \in [0, 1]$}
};
\node[anchor=west, align=left] at (\xclient+0.3, -3.00) {
    \myred{Derive feasible chunk budget: $B_{i,t} = \lfloor c_i \cdot a_{i,t} \cdot n_{chunks} \rfloor$ \quad (where $n_{chunks} = 4$ parameter chunks)}
};

% =============================================================
% ROUND 1: DP-SGD Training & Post-Noise Energy Ranking
% =============================================================
\def\yRoneTop{-3.8}
\def\yRoneBot{-8.2}

\draw[decorate, decoration={brace, amplitude=6pt, mirror}, thick] 
    (\xbrace, \yRoneBot) -- (\xbrace, \yRoneTop)
    node[midway, left=9pt, align=right] {
        \textbf{\large Round 1:}\\[3pt]
        \textbf{DP-SGD Training}\\[2pt]
        \& \myred{\textbf{Energy Ranking}}
    };

% Client DP-SGD and Energy Ranking
\node[anchor=west, align=left] at (\xclient+0.3, -4.10) {
    Compute microbatch gradients $g_b = \nabla \ell(W^{(n)}; x_b)$ and clip: $g_b \leftarrow g_b \cdot \min\left(1, \frac{C}{\|g_b\|_2}\right)$
};
\node[anchor=west, align=left] at (\xclient+0.3, -4.65) {
    Add calibrated Gaussian noise: $\tilde{g}_i \leftarrow \sum_b g_b + \mathcal{N}(0, \sigma^2 C^2 I)$ \quad [satisfies $(\varepsilon, \delta)$-DP via RDP accountant]
};
\node[anchor=west, align=left] at (\xclient+0.3, -5.20) {
    Compute local parameter update: $\Delta \tilde{w}_i = -\eta \tilde{g}_i$
};
\node[anchor=west, align=left] at (\xclient+0.3, -5.75) {
    Derive sensitivity threshold $\tau_i = \sigma C$; construct binary mask $M_i[k] = \mathbb{I}(|\Delta \tilde{w}_i[k]| \ge \tau_i)$
};
\node[anchor=west, align=left] at (\xclient+0.3, -6.35) {
    \myred{Decompose parameter update $\Delta \tilde{w}_i$ into $n_{chunks}$ parameter chunks ($4096$ slots per TenSEAL CKKS ciphertext)}
};
\node[anchor=west, align=left] at (\xclient+0.3, -6.95) {
    \myred{Rank chunks by post-noise energy: $\text{score}_j = \sum_{k \in \text{chunk } j} (M_i[k] \cdot \Delta \tilde{w}_i[k])^2$}
};
\node[anchor=west, align=left] at (\xclient+0.3, -7.55) {
    \myred{Select top-$B_{i,t}$ priority chunks $\mathcal{S}_i \subset \{1,\dots,n_{chunks}\}$ \quad (DP Post-Processing Theorem: $\Delta \varepsilon = 0$ privacy loss)}
};

% =============================================================
% ROUND 2: Hybrid Selective Encryption & Upload
% =============================================================
\def\yRtwoTop{-8.5}
\def\yRtwoBot{-12.9}

\draw[decorate, decoration={brace, amplitude=6pt, mirror}, thick] 
    (\xbrace, \yRtwoBot) -- (\xbrace, \yRtwoTop)
    node[midway, left=9pt, align=right] {
        \textbf{\large Round 2:}\\[3pt]
        \myred{\textbf{Hybrid Selective}}\\[2pt]
        \textbf{Encryption \& Upload}
    };

% Client HE action
\node[anchor=west, align=left] at (\xclient+0.3, -8.80) {
    Encrypt high-sensitivity coordinates under server public key $pk$:
};
\node[anchor=west, align=left] at (\xclient+0.3, -9.35) {
    \myred{Selectively encrypt top-$B_{i,t}$ chunks via CKKS: $c_i^{(j)} = \text{FHE.Enc}(\text{chunk}_j; pk) \quad \forall j \in \mathcal{S}_i$}
};
\node[anchor=west, align=left] at (\xclient+0.3, -9.90) {
    \myred{Leave residual $(n_{chunks} - B_{i,t})$ chunks in plaintext (provably protected by DP Gaussian noise floor)}
};

% Arrow Client -> Server
\draw[->, thick] (\xclient, -10.95) -- (\xserver, -10.95)
    node[midway, above=3pt] {\textbf{Send hybrid payload: $\langle \myred{\{c_i^{(j)}\}_{j \in \mathcal{S}_i}}, \myred{\{\text{chunk}_j\}_{j \notin \mathcal{S}_i}}, M_i, \myred{\mathcal{S}_i} \rangle$}};

% Server side action
\node[anchor=east, align=right] at (\xserver-0.3, -11.65) {
    Wait for updates from all participating clients $i \in \mathcal{C}_n$
};
\node[anchor=east, align=right] at (\xserver-0.3, -12.20) {
    \myred{Validate chunk budget compliance: verify $|\mathcal{S}_i| \le B_{i,t}$}
};

% =============================================================
% ROUND 3: Dual Aggregation & Active MitM Validation
% =============================================================
\def\yRthreeTop{-13.2}
\def\yRthreeBot{-17.4}

\draw[decorate, decoration={brace, amplitude=6pt, mirror}, thick] 
    (\xbrace, \yRthreeBot) -- (\xbrace, \yRthreeTop)
    node[midway, left=9pt, align=right] {
        \myred{\textbf{\large Round 3:}}\\[3pt]
        \myred{\textbf{Dual Aggregation}}\\[2pt]
        \myred{\textbf{\& MitM Defense}}
    };

% Server side aggregation
\node[anchor=east, align=right] at (\xserver-0.3, -13.55) {
    Aggregate parameter updates across vehicular clients:
};
\node[anchor=east, align=right] at (\xserver-0.3, -14.10) {
    Homomorphic CKKS aggregation for encrypted components: $C_{agg} = \boxplus_{i \in \mathcal{C}_n} c_i$
};
\node[anchor=east, align=right] at (\xserver-0.3, -14.70) {
    \myred{Dual aggregation per chunk $j$: $C_{agg}^{(j)} = \boxplus_{i: j \in \mathcal{S}_i} c_i^{(j)}$ (HE) \ and \ $P_{agg}^{(j)} = \sum_{i: j \notin \mathcal{S}_i} \text{chunk}_j$ (Plaintext)}
};
\node[anchor=east, align=right] at (\xserver-0.3, -15.35) {
    \myred{Active MitM Validation: Evaluate clean probe loss $L^{(n)}$ over held-out clean probe set}
};
\node[anchor=east, align=right] at (\xserver-0.3, -15.95) {
    \myred{3-$\sigma$ Anomaly Rule: If $|L^{(n)} - \mu_L| > 3 s_L \implies \text{flag tampering, discard } C_{agg}, \text{ roll back to } W^{(n-1)}$}
};
\node[anchor=east, align=right] at (\xserver-0.3, -16.55) {
    \myred{Else accept validated round updates and proceed to unmasking}
};

% =============================================================
% ROUND 4: Global Unmasking & Straggler-Free Sync
% =============================================================
\def\yRfourTop{-17.7}
\def\yRfourBot{-22.3}

\draw[decorate, decoration={brace, amplitude=6pt, mirror}, thick] 
    (\xbrace, \yRfourBot) -- (\xbrace, \yRfourTop)
    node[midway, left=9pt, align=right] {
        \textbf{\large Round 4:}\\[3pt]
        \textbf{Global Unmasking}\\[2pt]
        \& \myred{\textbf{Straggler Sync}}
    };

% Arrow Server -> Client
\draw[->, thick] (\xserver, -18.15) -- (\xclient, -18.15)
    node[midway, above=3pt] {\textbf{Broadcast aggregated updates: $C_{agg}$ and \myred{$P_{agg}$} to all clients}};

% Client local decryption and update
\node[anchor=west, align=left] at (\xclient+0.3, -18.80) {
    Each vehicle decrypts encrypted coordinates: $\Delta W_{enc} \leftarrow \text{Dec}_{sk_i}(C_{agg})$
};
\node[anchor=west, align=left] at (\xclient+0.3, -19.40) {
    Recombine with plaintext residual \myred{$P_{agg}$} and update: $W^{(n+1)} \leftarrow W^{(n)} + \frac{1}{|\mathcal{C}_n|} \Delta W$
};

% Guarantees / Outcomes
\node[anchor=west, align=left] at (\xclient+0.3, -20.20) {
    \myred{\textbf{Zero Straggler Latency:} Completes round in $32.4\,\text{s}$ ($5.7\times$ faster than waiting for slowest ECU in \textit{sd\_wait})}
};
\node[anchor=west, align=left] at (\xclient+0.3, -20.80) {
    \myred{\textbf{100\% Participation Fairness:} Maintains fairness ratio $= 1.00$ (prevents dropping weak ECUs in \textit{sd\_exclude})}
};
\node[anchor=west, align=left] at (\xclient+0.3, -21.40) {
    \myred{\textbf{Robust Accuracy under Attack:} $94.20\%$ test accuracy under severe active MitM gradient tampering}
};

\end{tikzpicture}

\vspace{0.55cm}
\begin{minipage}{0.96\textwidth}
\textbf{Figure: High-level protocol view of SecureDrive-FL with Compute-Budgeted and Energy-Adaptive Extensions.} Black text and solid lines show the original SecureDrive-FL paper's baseline protocol (client DP-SGD, sensitivity thresholding, and homomorphic CKKS aggregation). \underline{\textcolor{algored}{\textbf{Red, underlined text}}} represents our proposed research modifications to overcome vehicular heterogeneity (dynamic vehicular budget profiling, DP-safe post-noise energy ranking, hybrid encrypted/plaintext upload, dual-domain aggregation, active 3-$\sigma$ MitM anomaly filtering, and straggler-free completion).
\end{minipage}

\end{center}
\end{document}
'''
    return tex

def main():
    tex_file = "protocol_diagram.tex"
    pdf_file = "protocol_diagram.pdf"
    png_prefix = "protocol_diagram_page"
    output_png = "protocol_diagram.png"

    # Write TeX source
    print("Writing TeX source...")
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(generate_tikz_source())

    # Compile with pdflatex
    print("Compiling with pdflatex...")
    cmd_pdf = ["pdflatex", "-interaction=nonstopmode", "--miktex-disable-installer", tex_file]
    res_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
    if res_pdf.returncode != 0:
        print("pdflatex failed! Log tail:")
        print(res_pdf.stdout[-1500:])
        sys.exit(1)
    print("pdflatex compilation successful.")

    # Render to 300 DPI PNG with pdftoppm
    print("Rendering to high-res PNG (300 DPI) with pdftoppm...")
    cmd_ppm = ["pdftoppm", "-png", "-r", "300", pdf_file, png_prefix]
    res_ppm = subprocess.run(cmd_ppm, capture_output=True, text=True)
    if res_ppm.returncode != 0:
        print("pdftoppm failed:", res_ppm.stderr)
        sys.exit(1)

    generated_png = png_prefix + "-1.png"
    if not os.path.exists(generated_png):
        print(f"Error: {generated_png} not found.")
        sys.exit(1)

    # Crop whitespace automatically using PIL
    print("Cropping whitespace with PIL...")
    im = Image.open(generated_png)
    gray = im.convert("L")
    bbox = gray.point(lambda p: 255 if p < 250 else 0).getbbox()
    if bbox:
        pad = 40
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(im.width, bbox[2] + pad)
        bottom = min(im.height, bbox[3] + pad)
        cropped = im.crop((left, top, right, bottom))
    else:
        cropped = im

    # Save final PNG
    target_locations = [
        "sdfl_code/protocol_architecture.png",
        "sdfl_code/images/protocol_architecture.png",
        "sdfl_code/github_repo/figures/protocol_architecture.png",
        "sdfl_code/algorithm_diagram.png"
    ]
    for loc in target_locations:
        os.makedirs(os.path.dirname(loc), exist_ok=True)
        cropped.save(loc, "PNG")
        print(f"Saved: {loc}")

    # Also save cropped PDF or copy original PDF
    import shutil
    shutil.copyfile(pdf_file, "sdfl_code/protocol_architecture.pdf")
    shutil.copyfile(pdf_file, "sdfl_code/github_repo/figures/protocol_architecture.pdf")
    print("Saved PDF copies.")

if __name__ == "__main__":
    main()

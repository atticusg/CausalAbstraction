% =============================================================
% Slot into Section 2.1 (Conceptual Belief Spaces)
% After the definition of P(Z) \cong \Delta^6.
% =============================================================

\paragraph{A conformal metric on the belief simplex.}
How should we measure the quality of a belief trajectory induced by steering? A good metric should capture two distinct failure modes of steering: (1) abrupt jumps between distant concepts (e.g., teleporting from Monday to Friday without passing through intermediate days), and (2) passage through incoherent states where the model's output is not meaningfully concentrated on any concept. Prior work on steering evaluation has typically treated these as separate axes requiring separate metrics. We show that a single geometric construction---a \emph{conformal Riemannian metric} on the probability simplex---captures both simultaneously.

The idea is simple: we start with a standard notion of distance between probability distributions, and then warp it so that regions of the simplex the model never naturally visits become expensive to traverse. This is directly analogous to Fermat's principle in optics, where light bends around regions of high refractive index. In our setting, the ``refractive index'' is high in regions of the simplex that correspond to incoherent model states, causing well-behaved belief trajectories to curve around them.

\textbf{Base metric.}
As our starting point, we use the Fisher--Rao metric $g_{FR}$, the canonical Riemannian metric on the probability simplex \citep{TODO_amari_or_cencov}. For categorical distributions, this metric has a clean geometric interpretation: the square-root map $p \mapsto 2\sqrt{p}$ isometrically embeds the simplex onto a patch of the unit sphere, and Fisher--Rao distance is simply arc length on that sphere:
\begin{equation}
    d_{FR}(p, q) = 2\arccos\!\left(\textstyle\sum_k \sqrt{p_k \, q_k}\right).
\end{equation}
An important property of $d_{FR}$ for our purposes is that it respects the curvature of the simplex: paths through the high-entropy interior (where mass is spread across many concepts) are genuinely longer than they appear in flat embeddings. This means that even before adding any notion of coherence, the base metric already partially penalizes diffuse intermediate states---a point we return to in Section~\ref{TODO}.

\textbf{Conformal cost function.}
We now define which regions of the simplex are ``natural'' for the model. Let $\{q_j\}_{j=1}^{N}$ be the set of output distributions the model produces on unperturbed forward passes across the task---that is, the distributions it generates without any activation-space intervention. We define a scalar cost function on the simplex:
\begin{equation}
    c(p) = \min_{j} \, d_{FR}(p, q_j),
\end{equation}
the Fisher--Rao distance from $p$ to the nearest natural output. This cost is low near distributions the model naturally produces and high in regions it never visits.

Why does this capture conceptual structure, rather than just proximity to data points? The key observation is that the model's errors are \emph{structured}: when the model is uncertain about which weekday is correct, it tends to confuse \emph{adjacent} days (e.g., assigning mass to both Tuesday and Wednesday), not distant ones (e.g., Tuesday and Saturday). As a result, the natural output distributions do not merely cluster around seven point masses on the simplex. They also populate the arcs between adjacent concept vertices, tracing out a low-cost pathway that reflects the cyclic structure of the conceptual space. The interior of the simplex---corresponding to distributions that spread mass across non-adjacent concepts---remains high-cost because the model rarely produces such outputs.

\textbf{The conformal belief metric.}
We combine the base metric and the cost function into a conformal Riemannian metric:
\begin{equation}
    \tilde{g}_p = c(p)^2 \cdot g_{FR,p}.
\end{equation}
The induced distance between two distributions is then the length of the cheapest path between them:
\begin{equation}\label{eq:conformal_distance}
    d_c(p, q) = \inf_{\gamma:\,[0,1]\to\Delta} \int_0^1 c(\gamma(t))\,\|\dot{\gamma}(t)\|_{FR}\,dt,
\end{equation}
where the infimum is over all smooth paths $\gamma$ on the simplex. Since $c > 0$ everywhere, $d_c$ is a valid metric. The effect of the conformal rescaling is that paths staying in low-cost regions (near natural outputs) are short, while paths cutting through high-cost regions (conceptual voids) are long---even if the underlying Fisher--Rao distance traversed is the same. In the weekday example, the geodesic under $d_c$ from Monday to Thursday follows the arc through Tuesday and Wednesday (low cost, populated by natural outputs), rather than cutting through the simplex interior (high cost, devoid of natural outputs).

% =============================================================
% Slot into Section 2.4 (Evaluating Discrete Trajectories)
% =============================================================

\paragraph{Conformal path length as a unified evaluation metric.}
Given a discrete belief trajectory $b^\pi_0, \ldots, b^\pi_K$ induced by an activation-space path $\pi$, we approximate the conformal path length of Eq.~\eqref{eq:conformal_distance} by
\begin{equation}\label{eq:path_length}
    L_c(\pi) = \sum_{i=0}^{K-1} c(b^\pi_i) \cdot d_{FR}(b^\pi_i, b^\pi_{i+1}).
\end{equation}
Each term in this sum is the product of two factors: $d_{FR}(b^\pi_i, b^\pi_{i+1})$ measures how much the distribution changes at step $i$ (penalizing large jumps), while $c(b^\pi_i)$ measures how far the distribution at step $i$ is from anything the model naturally produces (penalizing incoherent intermediate states). A path that makes small, steady steps through natural distributions---exactly the behavior we observe from geodesic steering---will have low $L_c$ on both counts.

This formulation has no free parameters beyond the choice of base metric (Fisher--Rao) and cost function (1-nearest-neighbor to natural outputs). This is a notable simplification over optimal transport formulations, which require choosing a transport order $p$, a penalty multiplier for non-concept mass, and a summation scheme---all of which meaningfully affect the results.

\textbf{Decomposing the two axes.}
While $L_c$ provides a unified evaluation, it is useful to understand which factor drives the difference between steering methods. To this end, we also report the two components separately. The base path length
\begin{equation}
    L_{FR}(\pi) = \sum_{i=0}^{K-1} d_{FR}(b^\pi_i, b^\pi_{i+1})
\end{equation}
measures how far the belief trajectory travels in Fisher--Rao geometry, independent of whether it passes through natural regions. The average conformal cost along the path,
\begin{equation}
    \bar{c}(\pi) = \frac{1}{K+1}\sum_{i=0}^{K} c(b^\pi_i),
\end{equation}
measures how close the trajectory stays to the model's natural output manifold on average, serving as a continuous generalization of the discrete coherence score in Eq.~\eqref{eq:coherence}. Together, these provide interpretable axes: $L_{FR}$ captures efficiency (how direct is the path?) and $\bar{c}$ captures coherence (how natural are the intermediate states?). The conformal path length $L_c$ captures both.

% =============================================================
% Treatment of non-concept mass.
% Can go in Section 2.1 after conformal metric, or in methods.
% =============================================================

\paragraph{Treatment of non-concept probability mass.}
The model's full output distribution lives on the vocabulary simplex $\Delta^{|V|-1}$. We project to an $(|\mathcal{Z}|+1)$-dimensional sub-simplex by retaining the probability on each concept token and aggregating all remaining mass into a single ``other'' bin: $p_{\mathrm{other}} = 1 - \sum_{z \in \mathcal{Z}} p(z)$. A natural concern is how to handle this extra dimension---in optimal transport formulations, one must choose an ad hoc penalty for transporting mass to and from the ``other'' bin, and this choice substantially affects results.

The conformal metric sidesteps this issue entirely. Since the model's natural output distributions concentrate mass on concept tokens (low $p_\mathrm{other}$), any distribution with substantial non-concept mass is automatically far from all natural outputs in Fisher--Rao distance, and thus incurs high conformal cost $c(p)$. The ``other'' bin is simply an eighth coordinate in the simplex, treated identically to the seven concept coordinates. The penalty for incoherence emerges from the model's own behavior rather than from an externally imposed parameter.

% =============================================================
% Parallel geometry paragraph.
% Could go near Figure 2 or in Section 2.2 / Discussion.
% =============================================================

\paragraph{Parallel geometry across spaces.}
The conformal metric reveals a structural parallel between activation space and belief space (Figure~\ref{fig:above_below}). In activation space, concept representations cluster near a low-dimensional manifold $\mathcal{M} \subseteq \mathbb{R}^n$; linear steering leaves this manifold, passing through sparsely populated regions that do not correspond to coherent concept representations. In belief space, the model's natural output distributions cluster near a low-dimensional submanifold of the probability simplex; the conformal cost $c(p)$ measures distance to this output manifold, and paths that leave it are expensive under $d_c$.

The correspondence is not merely analogical. The model's input--output map $f: \mathbb{R}^n \to \Delta$ sends the activation-space manifold $\mathcal{M}$ to the low-cost region of belief space, and sends off-manifold activations to the high-cost interior. Geodesic steering stays on $\mathcal{M}$ and therefore produces belief trajectories in the low-cost region (low $\bar{c}$); linear steering leaves $\mathcal{M}$ and produces trajectories that cut through conceptual voids (high $\bar{c}$). The two manifolds---one in representation space, one in belief space---are images of each other under $f$, and the conformal metric makes this alignment quantitatively precise.
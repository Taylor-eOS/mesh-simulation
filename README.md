#Learning Probabilistic Relay Behavior in LoRa Mesh Networks
##Abstract
Meshtastic currently rely on simple flooding heuristics such as hop limits, duplicate suppression, randomized retransmission delays, and fixed signal thresholds. These approaches are robust and decentralized, but largely context-blind. A node deep inside a dense urban cluster and a node acting as a sparse bridge at the edge of coverage may apply identical retransmission behavior despite occupying fundamentally different positions within the propagation dynamics of the mesh.
This paper proposes reframing retransmission as a learned probabilistic propagation problem. Nodes locally infer retransmission probability from observable telemetry such as neighbor density, signal quality, congestion conditions, and recent propagation behavior. A lightweight neural policy is trained offline against propagation simulations that evaluate network-wide coverage outcomes under varying relay behaviors.
The expensive optimization occurs entirely offline. Deployment requires only compact local inference suitable for constrained hardware such as ESP32 devices. No topology dissemination, route maintenance, centralized coordination, or internet connectivity is required during operation. The resulting system preserves the robustness and graceful degradation properties of conventional flooding while using airtime more selectively and efficiently.
##The Problem with Flooding
Anyone who has operated Meshtastic in a busy area has observed the tradeoff inherent to flooding. In dense deployments, large fractions of airtime are consumed by redundant retransmissions. Multiple nearby nodes repeat packets that downstream receivers already obtained through other paths. In sparse deployments the opposite failure appears: suppression heuristics prevent retransmissions that were actually necessary for coverage continuity.
The difficulty is that the usefulness of retransmitting depends on context the node does not possess. A node does not know whether neighboring nodes already covered a downstream region; whether it is the sole bridge toward a sparse cluster several hops away or merely one of many redundant relays inside an already saturated region.
Traditional networking addresses this through routing. Nodes establish paths, maintain topology information, and forward packets toward destinations. But LoRa environments are poorly suited to classical routing. Links are intermittent, asymmetric, noisy, and highly load-dependent. Topology changes faster than low-bandwidth networks can disseminate updates. Airtime spent advertising routes is not available for payload traffic. Neighboring nodes sleep, move, disappear, and reappear unpredictably.
Flooding remains attractive precisely because it avoids these problems. It is decentralized, topology-agnostic, resilient to failures, and degrades gracefully under network fragmentation. Its weakness is that retransmission decisions are crude.
The proposal here is giving nodes better local estimates of when retransmitting is actually useful.
##Framing the Question
Routing asks: where should this packet go next?
Flooding asks: should I suppress this retransmission?
The framing proposed here asks: how much does my retransmission improve network-wide propagation?
A retransmission that reaches downstream nodes nobody else would have reached is valuable. A retransmission that duplicates coverage already provided by several nearby neighbors is not. Packets still diffuse through the mesh opportunistically. The model only influences retransmission probability. The goal is improving propagation efficiency while preserving the robustness characteristics of flooding: no route tables, no topology synchronization, no centralized coordination, graceful degradation under failures, tolerance to intermittent links, and mobility.
##What Nodes Can Actually Observe
The architecture is constrained by what deployed LoRa devices can realistically know locally. Disseminating full topology state across the network is impractical under LoRa airtime limitations.
A node can however observe: received signal strength and link quality, neighbor density, duplicate packet counts, retransmission timing patterns, local airtime congestion, hop counts, packet reception frequency, whether nearby neighbors already appear to be retransmitting, condensed fingerprints of neighboring nodes.
These signals contain statistical information about propagation context.
Nodes consistently observing many strong duplicates are likely inside dense saturated regions. Nodes consistently observing sparse weak receptions are likely near propagation frontiers or acting as structural bridges. Nodes surrounded by many active neighbors are less likely to contribute unique downstream coverage than isolated nodes with few alternatives.
None of these signals individually reveal global topology. But together they provide statistical evidence about likely propagation utility.
The offline model learns relationships between local telemetry patterns and network-wide propagation outcomes observed during offline simulation.
##Offline Training
The central challenge is that retransmission utility is not directly observable locally. A node cannot independently determine how much additional network coverage its retransmission produced. That depends on the behavior of the entire network.
This is why training occurs against a global propagation simulator.
The simulator generates large numbers of propagation scenarios. The model outputs retransmission probabilities for all nodes. The simulator then evaluates resulting propagation behavior across the network.
Training optimizes directly against propagation outcomes such as: overall coverage, reception probability, propagation completeness, airtime consumption, retransmission redundancy, congestion pressure.
The model is learning statistical retransmission behavior that produces desirable global propagation dynamics, not routes or forwarding decisions.
The deployment model does not possess global topology knowledge. The topology only exists during offline optimization. The trained weights compress statistical relationships between locally observable telemetry and globally useful relay behavior.
##Learned Relay Policies
The deployed system is not storing persistent node identities or centralized relay assignments. Each node performs lightweight local inference from its current telemetry state.
Internally, the model develops latent representations that encode propagation context.
Deployment requires no: centralized orchestration, topology-aware fingerprint distribution, synchronized node identities, or persistent global state.
The deployed artifact is simply the trained policy network itself.
##The Propagation Field
The system can be understood as operating on a probabilistic propagation field rather than on explicit routes.
When a node retransmits, it extends coverage into surrounding regions. When it suppresses, it allows nearby relays to carry propagation instead. The collective behavior of many locally acting nodes shapes the resulting propagation field across the network.
Dense areas should suppress redundant propagation. Sparse bridge regions should relay more aggressively. Congestion conditions should dynamically alter retransmission likelihoods. The behavior emerges statistically from local inference.
This resembles diffusion systems or epidemic propagation models more than conventional networking.
The model is not learning deterministic forwarding paths. It is learning how local propagation conditions statistically relate to useful relay behavior.
##This Is Not Routing
The system does not attempt to maintain stable paths or identify destinations. It does not compute shortest routes, distribute topology maps, or synchronize forwarding tables.
No node needs to know: where distant nodes are or how many hops away they exist.
Nodes just estimate whether retransmitting is likely to improve propagation coverage given their local observations.
This preserves the robustness properties that make flooding attractive.
The proposal is therefore not “machine learning routing”. It is learned probabilistic flooding.
##Deployment Feasibility
The deployment requirements are modest.
Inference can be implemented as a small feedforward neural network operating on compact telemetry vectors. Once trained, the deployed model may occupy only kilobytes. Runtime inference consists primarily of several matrix multiplications and activation functions, well within the capabilities of ESP32-class hardware.
No internet connectivity is required during operation. No centralized inference servers are needed. Nodes operate independently after deployment.
Updated policy weights could be distributed through: firmware releases, optional mesh broadcasts, manual updates, or even manual entry.
If the model is absent or disabled, nodes simply revert to conventional flooding behavior.
The system is therefore additive rather than disruptive.
##Conclusion
Current Meshtastic flooding succeeds because it is simple, decentralized, and resilient. Its weakness is that retransmission behavior is largely context-blind, and hence highly redundant.  The waste reduces the amount of hops the system can permit.
The system proposed here preserves the robustness properties of flooding while replacing fixed heuristics with learned probabilistic relay behavior trained against actual propagation outcomes.
The expensive computation occurs entirely offline. Deployment requires only lightweight local inference from telemetry already observable by the node itself.
The central claim is that local propagation telemetry contains enough statistical information to make substantially better retransmission decisions than fixed heuristics. By training directly against network-wide propagation outcomes, the system learns how local conditions relate to useful global propagation behavior.
The network still floods. It simply floods more intelligently.
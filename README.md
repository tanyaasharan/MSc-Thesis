# MSc-Thesis
This thesis explores how narratives shape market outcomes, prices, and strategies using the Bristol Stock Exchange. PRDE traders are extended with RA-RD opinion dynamics and dynamic malleability, embedded in Watts–Strogatz and Klemm–Eguíluz networks, aligning with Shiller’s Narrative Economics.

Robert Shiller, a Nobel Prize-winning Yale economist, wrote Narrative Economics, where he argues that people often make investment decisions based on stories of success and failure that spread like “psychological contagions” rather than on rational analysis. While he does not strongly support this theory with data, looking back at market booms and busts—the dot-com bubble, housing crash, bitcoin, and AI you can see recurring patterns.

#### To explore this further, I developed a computational model to study how such narratives might influence these market dynamics.

Here's a breakdown of the methodology
##### 1. Built upon the Bristol Stock Exchange an agent-based financial market simulation.
##### 2. Implemented a cognitive layer with the help of opinion dynamics within adaptive trading strategies to model decision-making in traders.
##### 3. Integrated trader interactions with complex social structures, to model and compare scale-free vs. small-world behaviour.
##### 4. The weight assigned to social interactions (peer influence) and external market signals (news, economic indicators) was dynamically varied throughout the simulation.

Ultimately, the model allowed me to study how prevailing narratives shape the codes submitted by buyers and sellers, and how these shifts cascade into trading strategies and, in turn, the broader market dynamics and prices.


Here’s an analysis from my simulations. In this case, I introduced an exogenous shock by shifting the demand and supply schedules from (75, 125) to (60, 110), which triggered a clear price drop.

<img width="545" height="233" alt="Screenshot 2025-09-24 at 5 07 12 PM" src="https://github.com/user-attachments/assets/e7d0855c-d7df-49f5-be63-42adf2a028f7" />

<img width="583" height="274" alt="Screenshot 2025-09-24 at 5 06 31 PM" src="https://github.com/user-attachments/assets/3cdcf8f2-4838-40c7-a1e7-330d46a55208" />

<img width="516" height="198" alt="Screenshot 2025-09-24 at 5 03 23 PM" src="https://github.com/user-attachments/assets/7d87a944-5484-488d-a0f7-54b34b1ca287" />

##### If you’re interested, we can definitely connect to discuss this topic further, and you can also check out my presentation for a more detailed explanation, along with the code in this repository.

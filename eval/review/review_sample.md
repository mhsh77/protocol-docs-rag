# Eval question review

For each question, mark **OK**, **FIX** (say what), or **DROP**. Things to check:
is it a realistic developer question, is the reference answer correct per the linked
page, and for unanswerable ones: are you confident the docs really don't answer it?

## adversarial (20 in set, showing 5)

### q008 · expected: `correct_premise` · false_premise
**Q:** Why is a cosigner not required in UniswapX RFQ since users can rely on any auctioneer to give the best price?
**Reference:** The passage explains that a cosigner is actually necessary in UniswapX RFQ. Users sign an order that defines a price range, and without a trusted party a malicious auctioneer could exploit that range and give the worst possible price. The cosigner field exists so users can specify a trusted auctioneer who will execute the auction within the signed parameters, protecting them from worst‑case pricing. Hence, a cosigner is required, not optional.
- evidence ([source](https://developers.uniswap.org/docs/liquidity/uniswapx/concepts/uniswaprfq)): "The cosigner field allows users to designate an auctioneer they trust to run the auction within signed parameters and return an executable quote."

Verdict: 

### q019 · expected: `answer` · outdated_term
**Q:** Back in Uniswap v3 we used fixed fee tiers—what are these so‑called 'flex fees' in the latest version?
**Reference:** Dynamic fees in Uniswap v4 are a specific type of swap fee paid by swappers that directly accrue to liquidity providers, and they can adjust in real‑time, change per swap, and be set to any percentage.
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/concepts/dynamic-fees)): "Dynamic fees in Uniswap v4 are a specific type of swap fee paid by swappers that directly accrue to liquidity providers."

Verdict: 

### q018 · expected: `answer` · outdated_term
**Q:** Older Uniswap docs talked about 'exchange contracts' that held liquidity and executed swaps. What are they called in the current protocol?
**Reference:** They are now called the automated market maker (AMM) contracts, i.e., the pool contracts that hold liquidity and execute swaps.
- evidence ([source](https://developers.uniswap.org/docs/protocols/overview)): "At its core are the automated market maker (AMM) contracts, the pool contracts that hold liquidity and execute swaps."

Verdict: 

### q005 · expected: `correct_premise` · false_premise
**Q:** Why does the permit signature work without adding an `EIP712Domain` type when using signing libraries?
**Reference:** The premise is incorrect. Some signing libraries do not automatically include the `EIP712Domain` type, so the permit signature will be rejected unless you explicitly add that domain definition to the types object before signing. Create a `typesWithDomain` object that includes the `EIP712Domain` fields and use it when calling `_signTypedData`.
- evidence ([source](https://developers.uniswap.org/docs/liquidity/liquidity-provisioning-api/integration-guide)): "Some signing libraries require an explicit `EIP712Domain` type in the types object. Add it if your library does not inject it automatically:"

Verdict: 

### q012 · expected: `abstain` · other_protocol
**Q:** What is the maximum leverage for perpetual futures trading on Uniswap?

Verdict: 

## factual (26 in set, showing 5)

### q040 · expected: `answer`
**Q:** What tick spacing should be used for a pool with a 0.30% fee?
**Reference:** For a 0.30% fee (lpFee = 3000), the tick spacing is 60.
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/guides/create-pool)): "| 0.30% | 3000      | 60           |"
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/guides/create-pool)): "_tickSpacing_ is the granularity of the pool. Lower values are more precise but may be more expensive to trade on"

Verdict: 

### q036 · expected: `answer`
**Q:** What deadline condition can cause an on‑chain transaction to revert according to the LP API guide?
**Reference:** A transaction will revert if it is not broadcast before the deadline encoded in the calldata. The deadline must be met prior to broadcasting the transaction.
- evidence ([source](https://developers.uniswap.org/docs/liquidity/liquidity-provisioning-api/integration-guide)): "**Check deadline**: Transaction was not broadcast before the deadline encoded in the calldata"

Verdict: 

### q041 · expected: `answer`
**Q:** In the v3 liquidity density chart, what does the total locked liquidity shown in the tooltip represent, and how is it calculated?
**Reference:** The tooltip’s total locked liquidity shows the sum of positions in the currency locked at the selected Tick, and it is calculated as the maximum token output of a swap when crossing to the next Tick.
- evidence ([source](https://developers.uniswap.org/docs/sdks/v3/guides/managing-liquidity/active-liquidity)): "The total locked liqudity in the tooltip represents the sum of positions in the currency locked at the selected Tick."
- evidence ([source](https://developers.uniswap.org/docs/sdks/v3/guides/managing-liquidity/active-liquidity)): "It is calculated as the maximum token output of a swap when crossing to the next Tick."

Verdict: 

### q039 · expected: `answer`
**Q:** Which Uniswap version introduced the ability for owners to set a subscriber on their positions?
**Reference:** v4
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/concepts/v4-vs-v3)): "Only v4: Owners can now set a subscriber for their positions."

Verdict: 

### q023 · expected: `answer`
**Q:** When decreasing liquidity and both tokens need to be sent, which actions should be encoded in the transaction?
**Reference:** You should encode the DECREASE_LIQUIDITY action followed by the TAKE_PAIR action.
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/guides/managing-liquidity/decrease-liquidity)): "* _take pair_ - receives a pair of tokens, to decrease liquidity"
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/guides/managing-liquidity/decrease-liquidity)): "bytes memory actions = abi.encodePacked(uint8(Actions.DECREASE_LIQUIDITY), uint8(Actions.TAKE_PAIR));"

Verdict: 

## multi_section (20 in set, showing 5)

### q066 · expected: `answer`
**Q:** When the /quote endpoint returns routing="CHAINED", what steps should I follow to execute the trade?
**Reference:** Use the /plan endpoints: create a plan with POST /plan, advance each step with PATCH /plan, and poll GET /plan until all steps complete, as outlined in the chained actions flow.
- evidence ([source](https://developers.uniswap.org/docs/trading/swapping-api/chained-actions)): "From there, three `/plan` endpoints drive execution: `POST /plan` creates the plan, `PATCH /plan` advances it as each step completes, and `GET /plan` reports the current state."
- evidence ([source](https://developers.uniswap.org/docs/trading/swapping-api/integration-guide)): "When `/quote` returns "routing": "CHAINED" (a trade that crosses chains or needs more than one step), drive execution through the `/plan` endpoints instead."

Verdict: 

### q047 · expected: `answer`
**Q:** After deploying a strategy that inherits from LBPStrategyBase and runs a CCA auction, how is the Uniswap v4 liquidity pool actually initialized?
**Reference:** The inheriting contract creates the LP pool using the auction proceeds, and once the CCA auction finishes you call migrate() after the migrationBlock to initialize the Uniswap v4 pool and its positions.
- evidence ([source](https://developers.uniswap.org/docs/liquidity/liquidity-launchpad/concepts/liquidity-strategies)): "Create a Uniswap v4 LP pool with the proceeds from the auction"
- evidence ([source](https://developers.uniswap.org/docs/liquidity/liquidity-launchpad/overview)): "call `migrate()` after `migrationBlock` to initialize the Uniswap v4 pool and liquidity positions"

Verdict: 

### q062 · expected: `answer`
**Q:** What are the two phases of UniswapX's auction and quoting processes, and how do they combine to give temporary exclusivity to winning quoters while preventing quote gaming?
**Reference:** UniswapX first runs an RFQ phase that grants temporary exclusivity to the winning quoter, then falls back to an open Dutch auction if that filler fails; the quoting flow is split into indicative (pre‑signature) and hard (post‑signature) quotes, and because quoters cannot tell which is which they must treat every quote as hard, ensuring competitive pricing and preventing gaming.
- evidence ([source](https://developers.uniswap.org/docs/liquidity/uniswapx/concepts/auction-types)): "This approach grants temporary exclusivity to winning quoters, then falls back to an open Dutch auction if the exclusive filler does not settle."
- evidence ([source](https://developers.uniswap.org/docs/liquidity/uniswapx/concepts/uniswaprfq)): "Quoters cannot distinguish between indicative and hard quotes. As a result, they must always assume all quotes are hard and provide competitive prices."

Verdict: 

### q055 · expected: `answer`
**Q:** What is the advantage of using Subscribers in Uniswap v4 compared to staking in v3 in terms of asset risk?
**Reference:** In v4, owners can attach a subscriber contract that receives notifications about their position without transferring the ERC‑721 token, so the liquidity and underlying assets stay under the owner's control; in v3, staking required moving the ERC‑721 token to a contract, exposing the assets to risk of malicious behavior.
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/concepts/subscribers)): "Through notification logic, position owners do not need to risk their liquidity position and its underlying assets."
- evidence ([source](https://developers.uniswap.org/docs/protocols/v4/concepts/v4-vs-v3)): "Staking in v3 requires users to transfer their ERC-721 token to a contract, putting the underlying assets at risk for malicious behavior."

Verdict: 

### q054 · expected: `answer`
**Q:** Why is manipulating a pool's liquidity considered much cheaper than manipulating its price, and how does that impact the feasibility of attacks over time?
**Reference:** Price manipulation incurs arbitrage losses and fees every block, which become prohibitive for large pools and long durations, while liquidity manipulation mainly costs only gas, making it far cheaper and more feasible.
- evidence ([source](https://developers.uniswap.org/docs/protocols/v2/concepts/oracles)): "The cost of manipulating the price for a specific time period can be roughly estimated as the amount lost to arbitrage and fees every block for the entire period."
- evidence ([source](https://developers.uniswap.org/docs/sdks/v3/guides/price-oracle)): "The costs associated with manipulating/ changing the liquidity of a Pool are orders of magnitude smaller than with manipulating the price of the assets, as prices will be arbitraged for assets with more than one market."

Verdict: 

## numeric (19 in set, showing 5)

### q073 · expected: `answer`
**Q:** What is the maximum size allowed for the raw X-Agent-Info header value before it is considered malformed?
**Reference:** 1024 bytes
- evidence ([source](https://developers.uniswap.org/docs/trading/swapping-api/agent-attribution)): "The raw header value is larger than **1024 bytes**, measured on the raw value before parsing."

Verdict: 

### q082 · expected: `answer`
**Q:** If a pool's tick is 0 for 5 seconds and then 100 for 5 seconds, what is the tick accumulator value?
**Reference:** 50
- evidence ([source](https://developers.uniswap.org/docs/protocols/v3/concepts/price-oracles)): "If the current tick on pool A is 0 for 5 seconds, and 100 for 5 seconds, the tick accumulator will be 50."

Verdict: 

### q084 · expected: `answer`
**Q:** What is the Ethereum mainnet address of the ETH/USDC 0.3% Uniswap V3 pool?
**Reference:** 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8
- evidence ([source](https://developers.uniswap.org/docs/protocols/v3/deployments/v3-optimism-deployments)): "For example, here is the [ETH/USDC 0.3% pool](https://etherscan.io/address/0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8) on Ethereum mainnet."

Verdict: 

### q079 · expected: `answer`
**Q:** What value is set for sqrtPriceLimitX96 to deactivate the price limit in the exactInputSingle swaps?
**Reference:** 0
- evidence ([source](https://developers.uniswap.org/docs/protocols/v3/guides/flash-swaps/flash-callback)): "For this example, we will set it to 0, which makes the argument inactive."

Verdict: 

### q077 · expected: `answer`
**Q:** What is the deployed address of the UniswapV3Factory contract on the MegaETH network?
**Reference:** 0x3a5f0cd7d62452b7f899b2a5758bfa57be0de478
- evidence ([source](https://developers.uniswap.org/docs/protocols/v3/deployments/v3-megaeth-deployments)): "| [UniswapV3Factory](https://github.com/Uniswap/uniswap-v3-core/blob/v1.0.0/contracts/UniswapV3Factory.sol)            | [`0x3a5f0cd7d62452b7f899b2a5758bfa57be0de478`](https://megaeth.blockscout.com/address/0x3a5f0cd7d62452b7f899b2a5758bfa57be0de478) |"

Verdict: 

## unanswerable (27 in set, showing 5)

### q090 · expected: `abstain`
**Q:** Does the OpenZeppelin Uniswap Hooks library expose a method to retrieve the current hook execution order, and what is the default order?

Verdict: 

### q093 · expected: `abstain`
**Q:** In the Uniswap v4 Template quick start, is there a default gas limit set for deploying the template contracts, and if so, what is it?

Verdict: 

### q106 · expected: `abstain`
**Q:** Does the DutchV3 exclusive auction have a minimum decay interval, and if so, what is its default value?

Verdict: 

### q102 · expected: `abstain`
**Q:** What is the estimated gas cost for creating a Uniswap v4 pool with 1000 ETH liquidity?

Verdict: 

### q098 · expected: `abstain`
**Q:** For the Uniswap v4 Community SDK Packages, is there a built-in retry mechanism for failed RPC calls, and how many retries does it perform by default?

Verdict: 

/*
def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
 */

export function topologicalSort(feedDict: Map<any, any>): void {
  const inputNodes = [];
  for(const key of feedDict.keys()) {
    inputNodes.push(key);
  }
  const G : any = {};
  const nodes = [...inputNodes];
  while(nodes.length > 0) {
    const n = nodes.pop();
    if(!G.hasOwnProperty(n)) {
      G[n] = {in: new Set(), out: new Set()};
    }
    for(const m of n.outboundNodes) {
      if(!G.hasOwnProperty(m)) {
        G[m] = {in:new Set(), out: new Set()};
        (G[n].in as Set<any>).add(m);
        (G[m].in as Set<any>).add(n);
        nodes.push(m);
      }
    }
  }
  const L = [];
  const S = new Set(inputNodes);
  while(S.size > 0) {
    const n = S.
  }
}

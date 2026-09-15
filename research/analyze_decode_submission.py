"""Align research signposts with actual target-process GPU active intervals."""
import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def table(path):
    root=ET.parse(path)
    refs={e.get("id"):e for e in root.iter() if e.get("id")}
    def resolve(e):
        return refs[e.get("ref")] if e.get("ref") else e
    return root,resolve


def merge(spans):
    out=[]
    for a,b in sorted(spans):
        if b<=a:
            continue
        if out and a<=out[-1][1]:
            out[-1][1]=max(b,out[-1][1])
        else:
            out.append([a,b])
    return out


def intersection(a,b):
    return merge((max(x,u),min(y,v)) for x,y in a for u,v in b if max(x,u)<min(y,v))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    base=args.directory
    data=json.loads((base/"submit-observer/result.json").read_text())
    root,resolve=table(base/"submit-signposts.xml")
    events={}
    offsets=[]
    for row in root.findall(".//row"):
        c=list(row)
        process=resolve(c[2])
        if int(resolve(process.find("pid")).text)!=data["pid"]:
            continue
        if resolve(c[6]).text!="DecodeProbe":
            continue
        metadata=resolve(c[11])
        label=resolve(metadata.find("string")).text
        timestamp=int(resolve(c[0]).text)
        native=int(resolve(metadata.find("uint64")).text)
        events[label]=timestamp
        offsets.append(timestamp-native)
    assert len(events)==len(data["events"]),"signpost event loss"
    root,resolve=table(base/"submit-gpu.xml")
    gpu=[]
    for row in root.findall(".//row"):
        c=list(row)
        p=resolve(c[10]).find("pid")
        if p is None or int(resolve(p).text)!=data["pid"] or resolve(c[7]).text!="Active":
            continue
        start=int(resolve(c[0]).text)
        gpu.append(dict(start_ns=start,end_ns=start+int(resolve(c[1]).text),
                        channel=resolve(c[2]).text,label=resolve(c[6]).get("fmt")))
    assert gpu,"target GPU intervals missing"
    with (base/"submit-target-gpu.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(gpu[0]));writer.writeheader();writer.writerows(gpu)
    rows=[]
    for sample in data["rows"]:
        tag=sample["tag"]
        lo,hi=events[tag+":layer-start"],events[tag+":layer-done"]
        spans=merge((max(lo,r["start_ns"]),min(hi,r["end_ns"])) for r in gpu if r["end_ns"]>lo and r["start_ns"]<hi)
        gaps=[];end=lo
        for a,b in spans:
            if a>end:gaps.append([end,a])
            end=b
        if end<hi:gaps.append([end,hi])
        ready_windows=[];graph_windows=[];expert_rows=[]
        for expert in data["selected"]:
            ready=events[f"{tag}:ready:{expert}"]
            available=max(lo,events.get(f"{tag}:read-done:{expert}",lo))
            submit=events[f"{tag}:submit-start:{expert}"]
            submitted=events[f"{tag}:submit-end:{expert}"]
            ready_windows.append([available,submit])
            graph_windows.append([ready,submit])
            expert_rows.append(dict(expert=expert,available_ns=available,yield_ns=ready,
                                    submit_ns=submit,submitted_ns=submitted))
        overlap=intersection(merge(ready_windows),gaps)
        # Exclude initial fill and final completion acknowledgement for a tighter ceiling.
        interior=intersection(overlap,[[spans[0][0],spans[-1][1]]])
        rows.append(dict(tag=tag,resident=sample["resident"],layer_ns=hi-lo,
                         gpu_active_union_ns=sum(b-a for a,b in spans),
                         no_target_gpu_activity_ns=sum(b-a for a,b in gaps),
                         ready_not_submitted_overlap_ns=sum(b-a for a,b in overlap),
                         interior_ready_gap_upper_ns=sum(b-a for a,b in interior),
                         post_yield_ready_to_submit_overlap_ns=sum(b-a for a,b in intersection(graph_windows,gaps)),
                         gpu_intervals=sum(lo<r["end_ns"] and r["start_ns"]<hi for r in gpu),
                         experts=expert_rows,gpu_spans=spans,gaps=gaps,ready_gap_spans=overlap))
    summary={str(n):{field:statistics.median(r[field] for r in rows if r["resident"]==n)/1e6
                    for field in ("layer_ns","gpu_active_union_ns","no_target_gpu_activity_ns",
                                  "ready_not_submitted_overlap_ns","interior_ready_gap_upper_ns")}
             for n in (10,4,0)}
    output=dict(evidence_kind="instrumented_fixed_layer_timeline",target_pid=data["pid"],
                aligned_events=len(events),target_gpu_intervals=len(gpu),
                clock_offset_spread_ns=max(offsets)-min(offsets),
                clock_offset_median_ns=statistics.median(offsets),medians_ms=summary,rows=rows,
                limits=["No activity means this target process only, not whole-device idle.",
                        "Overlap is an optimistic ceiling including necessary CPU graph building, not proven removable latency.",
                        "Native intervals identify command buffers, not individual kernels or expert last-use.",
                        "Observer timings are not uninstrumented performance results."])
    (base/"submit-analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k!="rows"},indent=2))


if __name__=="__main__":main()

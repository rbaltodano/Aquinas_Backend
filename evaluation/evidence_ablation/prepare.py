"""Build reproducible test-only A/B/C evidence inputs; never modifies corpus/app assets."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_retrieval as retrieval

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
# Whole, manually inspected corpus chunks, not generated answers. These indices
# refer to the recorded 50,283-passage export; the hash prevents silent remapping.
SELECTION = {
 'history-nicaea': [48122],
 'history-constantinople': [48549,48558],
 'history-nicaea-ii': [49707,49708],
 'history-chalcedon': [49932],
 'history-arius': [48296],
 'history-apostles-creed': [49901,49902],
 'history-nicene-creed': [48558],
 'scripture-john14': [21545,21546,21547],
 'scripture-lords-prayer': [22183],
 'scripture-good-samaritan': [22066],
 'scripture-resurrection': [22294],
 'doctrine-natural-law': [5817],
 'doctrine-virtue-prudence': [8102,8103],
 'doctrine-soul': [2112,2113],
 'moral-lying': [9729],
}

def main():
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--cpu', action='store_true', help='Use experimental simulator CPU backend, with a separate results file')
 args_cli = parser.parse_args()
 passages, embeddings = retrieval.load_corpus()
 assert len(passages) == 50283
 assert hashlib.sha256((retrieval.GROUNDING_DIR/'passages.json').read_bytes()).hexdigest() == '33b9c603276a2bf817cf721188d68e92910d7032b0f633ca1f4ae113e30f3ae4'
 cases = json.loads(retrieval.CASES.read_text())
 cases.append({'id':'scripture-john14','question':'What does John chapter 14 say?','category':'scripture'})
 selection = dict(SELECTION)
 chosen = [c for c in cases if c['id'] in selection or c['id'] in ['abstain-current-pope','abstain-vatican-ii']]
 assert len(chosen) == 17
 embed = retrieval.load_embedder()
 args = dict(embed=embed, passages=passages, embeddings=embeddings,
             chapters=retrieval.index_chapters(passages),book_aliases=retrieval.parse_book_aliases(),
             named_passages=retrieval.parse_named_passages(),curated=None,limit=3,floor=.55,corroboration_floor=.62)
 jobs=[]
 for n,c in enumerate(chosen):
  automatic=retrieval.retrieve(c['question'],**args)
  verified=[passages[i] for i in selection.get(c['id'],[])]
  evidence={'retrieved':automatic,'verified':verified,'none':[]}
  # No oracle exists for the missing-evidence cases; compare A versus C only.
  modes=['retrieved','verified','none'] if verified else ['retrieved','none']
  modes=modes[n%len(modes):]+modes[:n%len(modes)]
  for mode in modes:
   refs=[{'title':p['title'],'text':p['text']} for p in evidence[mode]]
   jobs.append({'id':c['id']+'--'+mode,'caseID':c['id'],'condition':mode,'question':c['question'],'references':refs})
 model=ROOT.parent/'Aquinas-iOS/Aquinas-iOS/LocalModels/gemma-4-E2B-it.litertlm'
 model_hash=hashlib.sha256(model.read_bytes()).hexdigest()
 assert model_hash=='9a6345f1a6cd39283f957977c84d31cc63b8dd56f2b8fffeb784940f63365282'
 suffix = '-cpu' if args_cli.cpu else ''
 manifest={'modelPath':str(model),'modelSHA256':model_hash,'useCPU':args_cli.cpu,'outputPath':str(OUT/f'answers{suffix}.json'),
           'corpusSHA256':hashlib.sha256((retrieval.GROUNDING_DIR/'passages.json').read_bytes()).hexdigest(),'jobs':jobs}
 (OUT/f'inputs{suffix}.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 (OUT/'verified-selection.json').write_text(json.dumps({c['id']:[dict(index=i,**passages[i]) for i in selection.get(c['id'],[])] for c in chosen},indent=2,ensure_ascii=False)+'\n')
 marker=Path('/tmp/aquinas-evidence-ablation-request.json')
 marker.write_text(json.dumps(manifest))
 print(f'Prepared {len(jobs)} jobs; {len(chosen)} questions. Marker: {marker}')
if __name__=='__main__': main()

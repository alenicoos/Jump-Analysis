import { ensureArtifactToolWorkspace, importArtifactTool } from '/Users/pietrolimoni/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs';
const workspace='/Users/pietrolimoni/Desktop/Jump-Analysis/outputs/manual-presentation-part2';
await ensureArtifactToolWorkspace(workspace);
const artifact = await importArtifactTool(workspace);
const { FileBlob, PresentationFile } = artifact;
const pres = await PresentationFile.importPptx(await FileBlob.load('/Users/pietrolimoni/Desktop/PoliMi/03-Template-Tesi-di-Laurea-ITA.pptx'));
const slide = (Array.isArray(pres.slides.items) ? pres.slides.items : Array.from({length:pres.slides.count}, (_,i)=>pres.slides.getItem(i)))[1];
for (const name of ['add','elements','shapes','placeholders']) {
  const obj = slide[name];
  console.log('\n==', name, 'type', typeof obj);
  if (typeof obj === 'function') {
    console.log('fn length', obj.length);
  } else if (obj) {
    console.log('keys', Object.keys(obj).slice(0,50));
    console.log('proto keys', Object.getOwnPropertyNames(Object.getPrototypeOf(obj)).sort().slice(0,80));
  }
}
console.log('artifact exports sample', Object.keys(artifact).filter(k=>/Text|Shape|Slide|Presentation|Image|Table|Fragment/.test(k)).sort().slice(0,120));

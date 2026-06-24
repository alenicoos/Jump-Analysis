import { ensureArtifactToolWorkspace, importArtifactTool } from '/Users/pietrolimoni/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs';
const workspace='/Users/pietrolimoni/Desktop/Jump-Analysis/outputs/manual-presentation-part2';
await ensureArtifactToolWorkspace(workspace);
const { FileBlob, PresentationFile } = await importArtifactTool(workspace);
const pres = await PresentationFile.importPptx(await FileBlob.load('/Users/pietrolimoni/Desktop/PoliMi/03-Template-Tesi-di-Laurea-ITA.pptx'));
const slide = (Array.isArray(pres.slides.items) ? pres.slides.items : Array.from({length:pres.slides.count}, (_,i)=>pres.slides.getItem(i)))[1];
console.log('shapes.add source', String(slide.shapes.add).slice(0,1200));
console.log('\nplaceholders.add source', String(slide.placeholders.add).slice(0,1200));
console.log('\nslide.add source', String(slide.add).slice(0,1200));

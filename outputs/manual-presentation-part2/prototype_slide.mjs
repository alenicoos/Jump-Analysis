import fs from 'node:fs/promises';
import path from 'node:path';
import { ensureArtifactToolWorkspace, importArtifactTool } from '/Users/pietrolimoni/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs';

const workspace='/Users/pietrolimoni/Desktop/Jump-Analysis/outputs/manual-presentation-part2/prototype';
await ensureArtifactToolWorkspace(workspace);
const artifact = await importArtifactTool(workspace);
const { FileBlob, PresentationFile } = artifact;
const src='/Users/pietrolimoni/Desktop/PoliMi/03-Template-Tesi-di-Laurea-ITA.pptx';
const pres = await PresentationFile.importPptx(await FileBlob.load(src));
const slides = Array.isArray(pres.slides.items) ? pres.slides.items : Array.from({length:pres.slides.count}, (_,i)=>pres.slides.getItem(i));
const content = slides[1].duplicate();
slides[0].delete();
slides[1].delete();
content.moveTo(0);
const contentSlide = (Array.isArray(pres.slides.items) ? pres.slides.items : Array.from({length:pres.slides.count}, (_,i)=>pres.slides.getItem(i)))[0];

const shapes = contentSlide.shapes.items;
for (const sh of shapes) {
  console.log('shape', sh.name, sh.id, sh.placeholderType, sh.isPlaceholder, sh.text ? String(sh.text) : 'no-text');
}
const title = shapes.find(s => s.name === 'Title 2');
if (title) {
  title.text = 'Sensors and Synchronization';
}
contentSlide.shapes.add({
  geometry: 'rect',
  position: { x: 90, y: 145, width: 1080, height: 380 },
  fill: 'none',
  line: 'none',
  text: 'Two BWT901CL IMUs near the knees\nGoal: synchronized video + IMU ground truth\nValid trials only',
  textStyle: {
    fontSize: 24,
    typeface: 'Calibri',
    color: '#1f1f1f',
    alignment: 'left',
    anchor: 1,
    wrap: 'square'
  }
});

await fs.mkdir(workspace, { recursive: true });
const preview = await pres.export({ slide: contentSlide, format: 'png', scale: 1 });
await fs.writeFile(path.join(workspace, 'prototype.png'), Buffer.from(await preview.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(pres);
await pptx.save(path.join(workspace, 'prototype.pptx'));
console.log('done');

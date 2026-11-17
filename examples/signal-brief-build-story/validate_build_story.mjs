import { createHash } from 'node:crypto';
import { access, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const root = dirname(fileURLToPath(import.meta.url));
const slug = 'signal-brief-build-story';
const markdown = await readFile(resolve(root, `${slug}.md`), 'utf8');
const html = await readFile(resolve(root, `${slug}.html`), 'utf8');
const snapshot = JSON.parse(await readFile(resolve(root, 'editorial-snapshot.json'), 'utf8'));
const ordered = value => Array.isArray(value) ? value.map(ordered) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, ordered(value[key])])) : value;
const { content_sha256, ...data } = snapshot;
const digest = createHash('sha256').update(JSON.stringify(ordered(data))).digest('hex');
if (digest !== content_sha256) throw new Error('Reviewed snapshot content hash mismatch');
const rendered = marked.parse(markdown, { gfm: true, breaks: false });
if (!html.includes(`<article>${rendered}</article>`)) throw new Error('HTML is stale relative to Markdown');
const prose = rendered.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const body = prose.split('Evidence and provenance')[0];
const numericWords = {zero:0,one:1,two:2,three:3,four:4,five:5};
const number = value => numericWords[value.toLowerCase()] ?? Number(value.replaceAll(',',''));
const factChecks = [];
function checkDisplayed(label, pattern, expected) {
  for (const match of prose.matchAll(pattern)) {
    if (number(match[1]) !== expected) throw new Error(`${label}: ${match[1]} differs from reviewed ${expected}`);
    factChecks.push({label,value:expected});
  }
}
const f = snapshot.field_run;
checkDisplayed('cached items', /([\d,]+) cached evidence items/gi, f.evidence_items);
checkDisplayed('current candidates', /([\d,]+) (?:balanced )?current candidates/gi, f.bounded_candidates);
checkDisplayed('selected candidates', /([\d,]+) candidates across/gi, f.bounded_candidates);
checkDisplayed('evidence lanes', /\b(\d+|one|two|three|four|five) evidence lanes/gi, f.evidence_lanes);
checkDisplayed('ranked stories', /\b(\d+|one|two|three|four|five) ranked stories/gi, f.ranked_stories);
checkDisplayed('ideas', /\b(\d+|one|two|three|four|five) ideas\b/gi, f.opportunities);
checkDisplayed('new triage calls', /\b(\d+|zero|one|two) new triage calls/gi, f.triaged_new_items);
checkDisplayed('Pitch recommendations', /\b(\d+|one|two|three) Pitch\b/g, f.recommendations.pitch);
checkDisplayed('Investigate recommendations', /\b(\d+|one|two|three) Investigate\b/g, f.recommendations.investigate);
checkDisplayed('whole-run seconds', /(\d+\.\d+) seconds/g, f.duration_ms/1000);
// Approximate earlier receipt durations are checked separately from the exact final-run duration.
for (const id of markdown.match(/2026-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z-weekly/g) ?? []) {
  if (!snapshot.weekly_runs.some(run => run.id === id)) throw new Error(`Unknown historical receipt: ${id}`);
}
for (const size of prose.matchAll(/([\d,]+) characters/g)) {
  if (![...snapshot.weekly_runs.map(run => run.prompt_characters),50000].includes(number(size[1]))) throw new Error('Unknown prompt-size claim');
}
for (const model of prose.match(/gpt-oss-\d+b/g) ?? []) {
  if (![snapshot.pipeline.triage.model,snapshot.pipeline.synthesis.model].some(value => value.endsWith(model))) throw new Error(`Unknown model: ${model}`);
}
for (const pattern of [/WANDB_API_KEY/i,/ANTHROPIC_API_KEY/i,/\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{12,}/,/\/Users\//,/\/var\/folders\//,/\b(?:TODO|TBD|PLACEHOLDER)\b/]) {
  if (pattern.test(html)) throw new Error(`Private or unresolved content: ${pattern}`);
}
if (/assets\/article-(?:news|content|evidence)-mobile/.test(html)) throw new Error('Reconstructed UI is not allowed in this revision');
const targets = new Set([...html.matchAll(/(?:href|src|srcset)="([^"]+)"/g)].map(match => match[1]));
for (const match of html.matchAll(/url\(['"]?([^)'"\s]+)['"]?\)/g)) targets.add(match[1]);
const external = [];
const local = [];
for (const target of targets) {
  if (/^https?:/.test(target)) { new URL(target); external.push(target); continue; }
  if (target.startsWith('#')) continue;
  const path = resolve(root, decodeURIComponent(target.split('#')[0]));
  if (relative(root,path).startsWith('..')) throw new Error(`Asset escapes publication directory: ${target}`);
  await access(path);
  local.push(target);
}
for (const target of local.filter(value => value.endsWith('.svg'))) {
  const svg = await readFile(resolve(root,target),'utf8');
  if (!svg.includes('<title') || !svg.includes('<desc')) throw new Error(`${target} needs an accessible title and description`);
  for (const run of snapshot.weekly_runs) {
    for (const value of [`~${Math.round(run.prompt_characters/1000)}k`,`${Math.round(run.duration_ms/1000)} s`]) {
      if (!svg.includes(`>${value}<`)) throw new Error(`${target} missing reviewed value ${value}`);
    }
  }
}
const links = [];
if (process.argv.includes('--links')) {
  for (const target of external) {
    const url = new URL(target);
    const safe = `${url.origin}${url.pathname}`;
    try {
      const response = await fetch(target,{signal:AbortSignal.timeout(20000)});
      await response.body?.cancel();
      let authenticated;
      if (response.status===404 && url.hostname==='github.com') {
        const parts=url.pathname.split('/').filter(Boolean);
        const endpoint=`repos/${parts[0]}/${parts[1]}${parts[2]==='commit'?`/commits/${parts[3]}`:''}`;
        const {stdout}=await promisify(execFile)('gh',['api',endpoint,'--jq','.html_url'],{timeout:20000});
        if (stdout.trim()!==safe) throw new Error('Authenticated GitHub destination differs');
        authenticated=true;
      }
      links.push({url:safe,status:response.status,authenticated,scope:url.hostname==='github.com'?'authenticated repository':'endpoint reachability; not a visual access check'});
    } catch(error) {links.push({url:safe,error:error.message});}
  }
}
const linksVerified=process.argv.includes('--links')?links.every(link=>link.authenticated || (link.status>=200 && link.status<400)):null;
const result = {ok:linksVerified!==false,bodyWords:body.split(/\s+/).length,totalWords:prose.split(/\s+/).length,snapshotSha256:digest,factChecks,localAssets:local,externalLinks:external.map(value=>{const u=new URL(value);return u.origin+u.pathname;}),linkChecks:links,linksVerified,notes:['No word-count or figure-count quota is enforced.','Historical receipt-only access and screenshot dates are disclosed in prose.']};
await writeFile(resolve(root,`${slug}-validation.json`),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
if(linksVerified===false)process.exitCode=1;

// Explicit official-client gate against the owned CPU fixture, never a model/API service.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { setTimeout as sleep } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

const [modulePath, baseURL, mode] = process.argv.slice(2);
const address = new URL(baseURL);
assert.equal(address.protocol, 'http:');
assert.ok(['127.0.0.1', '[::1]'].includes(address.hostname));
const watchdog = setTimeout(() => {
  console.error('SDK gate exceeded 10 seconds');
  process.exit(1);
}, 10_000);

try {
  const moduleURL = pathToFileURL(modulePath);
  const { default: OpenAI, BadRequestError } = await import(moduleURL.href);
  const pkg = JSON.parse(await readFile(new URL('./package.json', moduleURL), 'utf8'));
  assert.equal(pkg.name, 'openai');
  const client = new OpenAI({ apiKey: 'fixture', baseURL, maxRetries: 0, timeout: 2_000, logLevel: 'off' });
  const common = { model: 'ribn', temperature: 0 };
  const report = { mode, sdk_version: pkg.version, node_version: process.version };

  async function drain(stream, chat = false) {
    let text = '';
    let usage;
    const finishes = [];
    const roles = [];
    for await (const chunk of stream) {
      assert.equal(chunk.model, 'ribn');
      assert.ok(typeof chunk.id === 'string' && chunk.id.length > 0);
      assert.ok(Number.isSafeInteger(chunk.created));
      if (chunk.usage) {
        assert.equal(usage, undefined);
        assert.equal(finishes.length, 1);
        assert.deepEqual(chunk.choices, []);
        usage = chunk.usage;
      }
      for (const choice of chunk.choices) {
        assert.equal(choice.index, 0);
        assert.equal(usage, undefined);
        text += chat ? (choice.delta.content ?? '') : choice.text;
        if (chat && choice.delta.role) roles.push(choice.delta.role);
        if (choice.finish_reason != null) finishes.push(choice.finish_reason);
      }
    }
    assert.ok(usage);
    return { text, usage, finishes, roles };
  }

  if (mode === 'protocol') {
    const plain = await client.completions.create({ ...common, prompt: 'a', max_tokens: 2 });
    assert.equal(plain.choices[0].text, 'xx');
    assert.equal(plain.choices[0].finish_reason, 'length');
    assert.deepEqual(plain.usage, { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 });

    const streamed = await drain(await client.completions.create({ ...common, prompt: 'a', max_tokens: 3, stream: true, stream_options: { include_usage: true } }));
    assert.equal(streamed.text, 'xxx');
    assert.deepEqual(streamed.finishes, ['length']);
    assert.deepEqual(streamed.usage, { prompt_tokens: 1, completion_tokens: 3, total_tokens: 4 });

    // The fixture's 'e' prompt generates model EOS first. Usage counts that
    // committed hidden token, not visible text or the requested eight-token cap.
    const eos = await drain(await client.completions.create({ ...common, prompt: 'e', max_tokens: 8, stream: true, stream_options: { include_usage: true } }));
    assert.equal(eos.text, '');
    assert.deepEqual(eos.finishes, ['stop']);
    assert.deepEqual(eos.usage, { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 });

    const chat = await drain(await client.chat.completions.create({ ...common, messages: [{ role: 'user', content: 'a' }], max_tokens: 2, stream: true, stream_options: { include_usage: true } }), true);
    assert.equal(chat.text, 'xx');
    assert.deepEqual(chat.roles, ['assistant']);
    assert.deepEqual(chat.finishes, ['length']);
    assert.equal(chat.usage.completion_tokens, 2);
    assert.equal(chat.usage.total_tokens, chat.usage.prompt_tokens + 2);

    await assert.rejects(client.completions.create({ ...common, prompt: 'a', max_tokens: 1, temperature: 1 }), (error) => error instanceof BadRequestError && error.status === 400);
    const healthy = await client.completions.create({ ...common, prompt: 'a', max_tokens: 1 });
    assert.equal(healthy.choices[0].text, 'x');
    assert.equal(healthy.usage.completion_tokens, 1);
    report.requests = 6;
    report.hidden_eos_usage = eos.usage;
  } else if (mode === 'abort') {
    const stream = await client.completions.create({ ...common, prompt: 'a', max_tokens: 50, stream: true });
    let first = '';
    for await (const chunk of stream) {
      first += chunk.choices[0]?.text ?? '';
      if (first) break; // The SDK, not process exit, owns request cancellation.
    }
    assert.equal(first, 'x');
    assert.equal(stream.controller.signal.aborted, true);
    const deadline = performance.now() + 2_500;
    let abandoned = false;
    while (performance.now() < deadline) {
      const response = await fetch(new URL('/metrics', baseURL), { signal: AbortSignal.timeout(500) });
      assert.equal(response.status, 200);
      if ((await response.text()).includes('ribn_requests_abandoned_total 1\n')) {
        abandoned = true;
        break;
      }
      await sleep(10);
    }
    assert.equal(abandoned, true, 'server must observe abandonment while this SDK process is alive');
    report.observed_abandoned_before_exit = true;
  } else if (mode === 'recover') {
    const healthy = await client.completions.create({ ...common, prompt: 'a', max_tokens: 2 });
    assert.equal(healthy.choices[0].text, 'xx');
    assert.equal(healthy.usage.completion_tokens, 2);
    report.recovered = true;
  } else {
    throw new Error(`Unknown SDK gate mode: ${mode}`);
  }
  console.log(JSON.stringify(report));
} finally {
  clearTimeout(watchdog);
}

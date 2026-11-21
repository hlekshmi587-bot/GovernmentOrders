/**
 * Automated Component & Normalization Tests for MarkdownViewer
 *
 * Validates:
 * 1. Headings & backslash-escaped headings: \##, \###, \#### unescaped properly.
 * 2. Bold & backslash-escaped bold: \**text\** unescaped properly.
 * 3. Inline code & backslash-escaped code: \`code\` unescaped properly.
 * 4. Callout markers: \> [!IMPORTANT] unescaped properly.
 * 5. Preservation of legitimate Windows paths:
 *    - D:\GovernmentOrders\dataset\orders\GO_27.pdf
 *    - C:\Users\ACER\AppData\Local
 *    MUST NOT have their backslashes stripped or corrupted!
 * 6. Preservation of LaTeX arrows: $\rightarrow$ MUST NOT be damaged!
 */

import assert from 'node:assert/strict';

// Test implementation of normalizeMarkdown mirroring frontend/src/components/search/MarkdownViewer.tsx
export function normalizeMarkdown(text) {
  if (!text) return '';
  return text
    // Normalize escaped headings at start of line: \## -> ##, \### -> ###
    .replace(/^\\(#{1,6}\s)/gm, '$1')
    // Normalize escaped bold/italic: \**text\** -> **text**, \*text\* -> *text*
    .replace(/\\(\*{1,2})(.*?)\\(\*{1,2})/g, '$1$2$3')
    // Normalize isolated escaped asterisks attached to words: \** -> **
    .replace(/(^|[^\\])\\(\*{1,2})/g, '$1$2')
    // Normalize escaped blockquotes at start of line: \> -> >
    .replace(/^\\(>\s*)/gm, '$1')
    // Normalize escaped inline code: \`code\` -> `code`
    .replace(/\\(`)(.*?)\\(`)/g, '$1$2$3');
}

console.log('--- Running MarkdownViewer Normalization Automated Tests ---');

// Test 1: Escaped Headings
{
  const input = '\\## Government Orders Analysis Report\n\\### 1. Indexed Orders\n\\#### Sub-heading';
  const output = normalizeMarkdown(input);
  assert.equal(
    output,
    '## Government Orders Analysis Report\n### 1. Indexed Orders\n#### Sub-heading',
    'Escaped headings must be normalized to standard markdown'
  );
  console.log('✓ Test 1 Passed: Escaped headings normalized (\\## -> ##)');
}

// Test 2: Escaped Bold & Code
{
  const input = '\\**Order:\\** \\`G.O.(P)No.27/2026/FIN\\` | \\**Department:\\** Finance';
  const output = normalizeMarkdown(input);
  assert.equal(
    output,
    '**Order:** `G.O.(P)No.27/2026/FIN` | **Department:** Finance',
    'Escaped bold and inline code must be normalized'
  );
  console.log('✓ Test 2 Passed: Escaped bold and code normalized (\\** -> **, \\` -> `)');
}

// Test 3: Escaped Callout Notice
{
  const input = '\\> [!IMPORTANT]\n\\> **Official Notice:** Legal effect remains unverified.';
  const output = normalizeMarkdown(input);
  assert.equal(
    output,
    '> [!IMPORTANT]\n> **Official Notice:** Legal effect remains unverified.',
    'Escaped blockquotes must be normalized'
  );
  console.log('✓ Test 3 Passed: Escaped callout notice normalized (\\> -> >)');
}

// Test 4: Preservation of Windows file paths
{
  const winPath1 = 'D:\\GovernmentOrders\\dataset\\orders\\GO_27.pdf';
  const winPath2 = 'C:\\Users\\ACER\\AppData\\Local\\Programs';
  assert.equal(normalizeMarkdown(winPath1), winPath1, 'Windows path 1 must be strictly preserved');
  assert.equal(normalizeMarkdown(winPath2), winPath2, 'Windows path 2 must be strictly preserved');
  console.log('✓ Test 4 Passed: Windows file paths preserved intact without stripping backslashes');
}

// Test 5: Preservation of LaTeX arrows
{
  const latexStr = '- Order A $\\rightarrow$ Action $\\rightarrow$ Order B';
  assert.equal(normalizeMarkdown(latexStr), latexStr, 'LaTeX arrow must be strictly preserved');
  console.log('✓ Test 5 Passed: LaTeX arrow ($\\rightarrow$) preserved intact');
}

console.log('--- All MarkdownViewer Normalization Tests Passed Successfully ---');

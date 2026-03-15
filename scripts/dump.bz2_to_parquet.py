"""
Nettoyage dump Wikipedia FR (bz2) → Parquet shardé.

Extrait le texte des articles (namespace 0) en conservant uniquement :
  - le texte brut
  - les wikilinks [[...]] et [[...|texte]]

Tout le reste est supprimé (html/tables/refs/etc).
Amélioration clé : certains templates inline (ex: {{japonais|...}}) sont "rendus"
en texte (extraction paramètre utile) au lieu d'être supprimés.

"""

import bz2
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count
from typing import Optional
import mwparserfromhell
import pyarrow as pa
import pyarrow.parquet as pq
import hashlib
import numpy as np
import pyarrow.feather as feather

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

DUMP_FILE   = 'frwiki-20260201-pages-articles-multistream.xml.bz2'
OUTPUT_DIR  = 'output_parquet'
SHARD_SIZE  = 50_000
NUM_WORKERS = max(1, cpu_count() - 1)
CHUNKSIZE   = 64
LOG_EVERY   = 1_000
LIMIT       = None
COMPRESSION = 'snappy'
RESUME_INDEX_FILE = "processed_titles_hash64.feather"


# Préfixes de wikilinks à supprimer (fichiers, catégories, espaces de noms, interwiki…)
PREFIXES_A_SUPPRIMER = [
    'fichier:', 'file:', 'image:', 'img:', 'média:', 'media:',
    'catégorie:', 'category:',
    'wikipédia:', 'wikipedia:', 'wp:',
    'aide:', 'help:',
    'portail:', 'projet:', 'project:',
    'modèle:', 'template:',
    'spécial:', 'special:',
    'module:',
    'discussion:', 'talk:',
    'utilisateur:', 'user:',
    'wikt:', 'wiktionnaire:', 'commons:',
    's:', 'wikisource:',
    'w:', 'n:', 'q:', 'b:', 'v:',
    'wikidata:', 'd:', 'meta:',
]

# Balises HTML dont on garde le contenu textuel
TAGS_GARDER_CONTENU = {
    'b', 'i', 'u', 'em', 'strong', 'small', 'big',
    'span', 'div', 'p', 'blockquote', 'center',
    'sub', 'sup', 'abbr', 'cite', 's', 'del', 'ins',
}

# Balises HTML remplacées par un saut de ligne
TAGS_SAUT_DE_LIGNE = {'br', 'hr'}


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA PARQUET (simplifié)
# ═════════════════════════════════════════════════════════════════════════════

PARQUET_SCHEMA = pa.schema([
    ('title', pa.string()),
    ('text',  pa.string()),
])


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═════════════════════════════════════════════════════════════════════════════

def log(msg=''):
    print(msg, flush=True)

def log_err(msg):
    print(msg, file=sys.stderr, flush=True)

def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def format_size(size_bytes):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size_bytes < 1024:
            return f'{size_bytes:.1f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.1f} PB'


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTION XML
# ═════════════════════════════════════════════════════════════════════════════

def iter_xml_pages(filepath):
    """
    Générateur bz2 xml → yield (page_id, title, text) pour ns=0.
    """
    with bz2.open(filepath, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=('start', 'end'))
        event, root = next(context)

        ns_uri = ''
        if '}' in root.tag:
            ns_uri = root.tag[:root.tag.index('}') + 1]

        page_tag     = f'{ns_uri}page'
        ns_tag       = f'{ns_uri}ns'
        title_tag    = f'{ns_uri}title'
        id_tag       = f'{ns_uri}id'
        revision_tag = f'{ns_uri}revision'
        text_tag     = f'{ns_uri}text'

        for event, elem in context:
            if event != 'end' or elem.tag != page_tag:
                continue

            ns_elem = elem.find(ns_tag)
            if ns_elem is None or ns_elem.text != '0':
                root.clear()
                continue

            title_elem = elem.find(title_tag)
            title = title_elem.text if title_elem is not None and title_elem.text else ''

            id_elem = elem.find(id_tag)
            page_id = int(id_elem.text) if id_elem is not None and id_elem.text else 0

            revision = elem.find(revision_tag)
            text_elem = revision.find(text_tag) if revision is not None else None
            text = text_elem.text if text_elem is not None and text_elem.text else ''

            root.clear()
            yield page_id, title, text


# ═════════════════════════════════════════════════════════════════════════════
# TEMPLATE "RENDERING" (inline templates utiles)
# ═════════════════════════════════════════════════════════════════════════════

# Templates typiques en début d’article qui affichent un nom en paramètre 1
# (ex: {{japonais|'''Chiba'''|...}} -> on veut juste param 1)
TEMPLATES_LANG_ADJ_PARAM1 = {
    'japonais', 'chinois', 'coréen', 'arabe', 'hébreu', 'russe', 'ukrainien',
    'grec', 'latin', 'allemand', 'anglais', 'espagnol', 'italien', 'portugais',
    'néerlandais', 'suédois', 'norvégien', 'danois', 'finnois', 'turc', 'persan',
    'polonais', 'tchèque', 'slovaque', 'hongrois', 'roumain', 'bulgare', 'serbe',
    'croate', 'slovène', 'estonien', 'letton', 'lituanien',
}

# Débuts d’intro “suspects” si le sujet a été mangé par un template supprimé
INTRO_START_WORDS = {
    'est', 'sont', 'fut', 'était', 'étaient', 'désigne', 'constitue',
    'représente', 'correspond', 'regroupe', 'comprend', 'fait', 'peut',
}

_LATIN_RE = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿ]')

def _normalize_template_name(name: str) -> str:
    name = name.strip().replace('_', ' ')
    low = name.lower()

    # {{Modèle:xxx}} ou {{Template:xxx}}
    if ':' in low:
        prefix, rest = low.split(':', 1)
        if prefix in {'modèle', 'template', 'model'}:
            low = rest.strip()
    return low

def _get_param_value(tpl, key):
    """
    Retourne le wikitexte du paramètre (positional int ou named str).
    """
    try:
        if tpl.has(key):
            return str(tpl.get(key).value).strip()
    except Exception:
        return ''
    return ''

def _contains_latin(s: str) -> bool:
    return bool(_LATIN_RE.search(s))

def resolve_template_to_wikitext(tpl) -> Optional[str]:
    """
    Retourne une chaîne (wikitexte) si le template doit être converti en texte,
    sinon None => template supprimé.
    """
    name = _normalize_template_name(str(tpl.name))

    # Parser function {{formatnum:12345}} (souvent sous forme de "nom:val")
    if name.startswith('formatnum:'):
        return name.split(':', 1)[1].strip()

    # Magic words / maintenance non textuelles
    if name.startswith('defaultsort:') or name.startswith('pagename'):
        return None

    # Language adjective templates: param 1 = texte FR utile (souvent le sujet en gras)
    if name in TEMPLATES_LANG_ADJ_PARAM1:
        v = _get_param_value(tpl, 1)
        return v if v else None

    # {{lang|code|texte}} / {{langue|code|texte}} : garder seulement si ça contient du latin
    if name in {'lang', 'langue'}:
        v = _get_param_value(tpl, 2) or _get_param_value(tpl, 1)
        if v and _contains_latin(v):
            return v
        return None

    # {{lang-xx|texte}} : garder seulement si latin (sinon bruit CJK/cyrillique, etc.)
    if name.startswith('lang-'):
        v = _get_param_value(tpl, 1)
        if v and _contains_latin(v):
            return v
        return None

    # {{lien|Texte|...}} : le texte affiché est généralement param 1
    if name == 'lien':
        v = _get_param_value(tpl, 1) or _get_param_value(tpl, 'texte')
        return v if v else None

    # Tout le reste: suppression (infobox, palettes, maintenance, etc.)
    return None

def maybe_prepend_title(title: str, cleaned: str) -> str:
    """
    Heuristique anti-'sujet supprimé':
    - on cherche le premier mot alphabétique
    - s'il est en minuscule ET dans un set de débuts d'intro typiques => prefixe le titre
    """
    if not title or not cleaned:
        return cleaned

    # évite doublons
    title_base = re.sub(r'\s*\(.*?\)\s*$', '', title).strip()
    if title_base and cleaned.lower().startswith(title_base.lower()):
        return cleaned

    m = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", cleaned)
    if not m:
        return cleaned

    first_word = m.group(0)
    if first_word and first_word[0].islower() and first_word.lower() in INTRO_START_WORDS:
        return f"{title_base} {cleaned}".strip()

    return cleaned


# ═════════════════════════════════════════════════════════════════════════════
# NETTOYAGE WIKITEXT
# ═════════════════════════════════════════════════════════════════════════════

def clean_wikitext(text: str, title: str = '') -> str:
    """
    Nettoie le wikicode en conservant les wikilinks [[...]] / [[...|...]].
    """
    wikicode = mwparserfromhell.parse(text)

    # 1) Commentaires <!-- ... -->
    for node in wikicode.filter_comments():
        try:
            wikicode.remove(node)
        except ValueError:
            pass

    # 2) Balises HTML
    for tag in wikicode.filter_tags():
        try:
            tag_name = str(tag.tag).lower().strip()
            if tag_name in TAGS_GARDER_CONTENU:
                wikicode.replace(tag, tag.contents if tag.contents else '')
            elif tag_name in TAGS_SAUT_DE_LIGNE:
                wikicode.replace(tag, '\n')
            else:
                wikicode.remove(tag)
        except ValueError:
            pass

    # 3) Templates: rendre ceux qui sont "inline text", supprimer les autres
    # Plusieurs passes pour gérer l'imbrication
    for _ in range(8):
        templates = wikicode.filter_templates(recursive=True)
        if not templates:
            break

        any_change = False
        for tpl in templates:
            try:
                repl = resolve_template_to_wikitext(tpl)
                if repl is None:
                    wikicode.remove(tpl)
                else:
                    # important: on parse repl pour exposer wikilinks/markup au reste du pipeline
                    wikicode.replace(tpl, mwparserfromhell.parse(repl))
                any_change = True
            except ValueError:
                pass

        if not any_change:
            break

    # 4) Wikilinks: supprimer certains namespaces
    for link in wikicode.filter_wikilinks():
        target = str(link.title).strip()
        if any(target.lower().startswith(p) for p in PREFIXES_A_SUPPRIMER):
            try:
                wikicode.remove(link)
            except ValueError:
                pass

    # 5) Liens externes [http://... texte]
    for extlink in wikicode.filter_external_links():
        try:
            if extlink.title:
                wikicode.replace(extlink, str(extlink.title))
            else:
                wikicode.remove(extlink)
        except ValueError:
            pass

    # 6) Titres de section == ==
    for heading in wikicode.filter_headings():
        try:
            wikicode.remove(heading)
        except ValueError:
            pass

    # 7) Post-traitement texte
    result = str(wikicode)

    # Tables wiki {| ... |}
    result = re.sub(r'\{\|.*?\|\}', '', result, flags=re.DOTALL)

    # Gras/italique
    result = result.replace("'''", '').replace("''", '')

    # Behavior switches __TOC__ etc.
    result = re.sub(r'__[A-Z_]+__', '', result)

    # Puces/listes début de ligne
    result = re.sub(r'^[\*#;:]+\s*', '', result, flags=re.MULTILINE)

    # Lignes horizontales ----
    result = re.sub(r'^-{4,}\s*$', '', result, flags=re.MULTILINE)

    # Fragments {{ }} résiduels
    result = re.sub(r'\{\{[^{}]*\}\}', '', result)

    # Entités HTML résiduelles
    result = result.replace('&nbsp;', ' ')
    result = re.sub(r'&[a-zA-Z]+;', '', result)
    result = re.sub(r'&#\d+;', '', result)

    # Espaces
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r' *\n *', '\n', result)
    result = re.sub(r'\n{3,}', '\n\n', result)

    # Lignes ponctuation orpheline
    result = re.sub(r'^\s*[,;.:\-–—|()]*\s*$', '', result, flags=re.MULTILINE)
    result = re.sub(r'\n{3,}', '\n\n', result)

    result = result.strip()
    result = maybe_prepend_title(title, result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# WORKER
# ═════════════════════════════════════════════════════════════════════════════

def clean_page(page):
    page_id  = page['id']
    title    = page['title']
    raw_text = page['text']

    try:
        cleaned = clean_wikitext(raw_text, title=title)
        return {
            'id':    page_id,   # uniquement pour logs erreurs
            'title': title,
            'text':  cleaned,
            'error': None,
        }
    except Exception as e:
        return {
            'id':    page_id,
            'title': title,
            'text':  '',
            'error': f'{type(e).__name__}: {e}',
        }


# ═════════════════════════════════════════════════════════════════════════════
# PARQUET
# ═════════════════════════════════════════════════════════════════════════════

def write_shard(records, shard_idx, output_dir):
    filepath = os.path.join(output_dir, f'wiki_shard_{shard_idx:05d}.parquet')

    data = {
        'title': [r['title'] for r in records],
        'text':  [r['text']  for r in records],
    }

    table = pa.table(data, schema=PARQUET_SCHEMA)
    pq.write_table(table, filepath, compression=COMPRESSION)

    file_size = os.path.getsize(filepath)
    return filepath, file_size


def title_hash64(title: str) -> int:
    d = hashlib.blake2b(title.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little", signed=False)

def load_resume_index(path: str):
    tbl = feather.read_table(path, columns=["title_hash64"])
    hashes = tbl.column(0).to_numpy(zero_copy_only=False)  # np.uint64

    # ensure sorted (should already be sorted)
    if len(hashes) > 1 and not np.all(hashes[:-1] <= hashes[1:]):
        hashes.sort()

    md = tbl.schema.metadata or {}
    next_idx = int((md.get(b"next_shard_idx") or b"0").decode("utf-8"))
    return hashes, next_idx

def is_processed(hashes_sorted: np.ndarray, title: str) -> bool:
    h = np.uint64(title_hash64(title))
    i = hashes_sorted.searchsorted(h)
    return i < hashes_sorted.size and hashes_sorted[i] == h

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # ── Check dump exists ──
    if not os.path.isfile(DUMP_FILE):
        log_err(f"ERREUR : dump introuvable")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Resume index (optional) ──
    resume_hashes = None
    shard_idx_start = 0

    if RESUME_INDEX_FILE and os.path.isfile(RESUME_INDEX_FILE):
        resume_hashes, shard_idx_start = load_resume_index(RESUME_INDEX_FILE)
        log(f"  Resume index loaded: {resume_hashes.size:,} titles")
        log(f"  Starting shard index: {shard_idx_start:05d}")
        log()

    # ── Stats ──
    stats = {
        'pages_xml':       0,
        'redirections':    0,
        'pages_vides':     0,
        'erreurs':         0,
        'pages_nettoyees': 0,
        'shards_ecrits':   0,
        'taille_totale':   0,
    }
    erreurs_detail = []

    t_start = time.time()

    log(f"{'═' * 80}")
    log("  NETTOYAGE DUMP WIKIPEDIA FR → PARQUET (title,text)")
    log(f"{'═' * 80}")
    log(f"  Fichier source  : {DUMP_FILE} ({format_size(os.path.getsize(DUMP_FILE))})")
    log(f"  Dossier sortie  : {OUTPUT_DIR}/")
    log(f"  Pages par shard : {SHARD_SIZE:,}")
    log(f"  Workers         : {NUM_WORKERS}")
    log(f"  Compression     : {COMPRESSION}")
    log(f"  Limite          : {'aucune' if LIMIT is None else f'{LIMIT:,} pages'}")
    log(f"{'═' * 80}")
    log()

    def article_generator():
        articles_yielded = 0
        for page_id, title, text in iter_xml_pages(DUMP_FILE):
            stats['pages_xml'] += 1

            if not text:
                stats['pages_vides'] += 1
                continue

            if re.match(r'#(redirect|redirection)', text, re.IGNORECASE):
                stats['redirections'] += 1
                continue

            # Skip already-processed titles (resume)
            if resume_hashes is not None and is_processed(resume_hashes, title):
                continue

            yield {'id': page_id, 'title': title, 'text': text}

            articles_yielded += 1
            if LIMIT is not None and articles_yielded >= LIMIT:
                break

    shard_buffer = []
    shard_idx = shard_idx_start

    if NUM_WORKERS > 1:
        pool = Pool(NUM_WORKERS)
        results_iter = pool.imap_unordered(clean_page, article_generator(), chunksize=CHUNKSIZE)
    else:
        results_iter = (clean_page(page) for page in article_generator())

    try:
        for result in results_iter:
            if result['error'] is not None:
                stats['erreurs'] += 1
                erreurs_detail.append((result['id'], result['title'], result['error']))
                if len(erreurs_detail) <= 20:
                    log_err(f"  ⚠ Erreur id={result['id']} « {result['title']} » : {result['error']}")
                continue

            shard_buffer.append(result)
            stats['pages_nettoyees'] += 1

            if stats['pages_nettoyees'] % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                speed = stats['pages_nettoyees'] / elapsed if elapsed > 0 else 0
                log(
                    f"  [{format_duration(elapsed)}] "
                    f"{stats['pages_nettoyees']:>10,} nettoyées "
                    f"({speed:>7.1f} p/s) │ "
                    f"XML lues: {stats['pages_xml']:>10,} │ "
                    f"redir: {stats['redirections']:>9,} │ "
                    f"err: {stats['erreurs']:>5} │ "
                    f"→ {result['title'][:45]}"
                )

            if len(shard_buffer) >= SHARD_SIZE:
                try:
                    filepath, fsize = write_shard(shard_buffer, shard_idx, OUTPUT_DIR)
                    stats['shards_ecrits'] += 1
                    stats['taille_totale'] += fsize
                    log(f"Shard {shard_idx:05d} écrit : {filepath} ({len(shard_buffer):,} pages, {format_size(fsize)})")
                except Exception as e:
                    log_err(f"  ❌ Erreur écriture shard {shard_idx:05d} : {e}")
                shard_buffer = []
                shard_idx += 1

    except KeyboardInterrupt:
        log()
        log("  ⚠ Interruption clavier (Ctrl+C) — écriture du shard partiel…")

    finally:
        if NUM_WORKERS > 1:
            pool.terminate()
            pool.join()

    if shard_buffer:
        try:
            filepath, fsize = write_shard(shard_buffer, shard_idx, OUTPUT_DIR)
            stats['shards_ecrits'] += 1
            stats['taille_totale'] += fsize
            log(f"Shard {shard_idx:05d} écrit : {filepath} ({len(shard_buffer):,} pages, {format_size(fsize)})")
        except Exception as e:
            log_err(f"  ❌ Erreur écriture shard final {shard_idx:05d} : {e}")

    elapsed = time.time() - t_start
    speed = stats['pages_nettoyees'] / elapsed if elapsed > 0 else 0

    log()
    log("  RÉSUMÉ")
    log(f"{'═' * 80}")
    log(f"  Pages XML (ns=0) lues    : {stats['pages_xml']:>12,}")
    log(f"  Redirections ignorées    : {stats['redirections']:>12,}")
    log(f"  Pages vides ignorées     : {stats['pages_vides']:>12,}")
    log(f"  Erreurs de nettoyage     : {stats['erreurs']:>12,}")
    log(f"  Pages nettoyées écrites  : {stats['pages_nettoyees']:>12,}")
    log(f"{'─' * 80}")
    log(f"  Shards Parquet créés     : {stats['shards_ecrits']:>12,}")
    log(f"  Taille totale sortie     : {format_size(stats['taille_totale']):>12}")
    log(f"  Durée totale             : {format_duration(elapsed):>12}")
    log(f"  Vitesse moyenne          : {speed:>11.1f} p/s")
    log(f"{'═' * 80}")

    if erreurs_detail:
        n = len(erreurs_detail)
        log()
        log(f"  ⚠ {n} erreur(s) rencontrée(s).")
        for pid, ptitle, perr in erreurs_detail[:50]:
            log(f"      id={pid:<10} {ptitle:<45} {perr}")
        if n > 50:
            log(f"      … et {n - 50} autres.")

    log()
    log("Terminé.")
    log()


if __name__ == '__main__':
    main()

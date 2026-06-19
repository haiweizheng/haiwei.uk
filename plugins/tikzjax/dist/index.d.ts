import { QuartzTransformerPlugin } from '@quartz-community/types';

type TexPackages = Record<string, string>;
interface TikzJaxOptions {
    texPackages?: TexPackages;
    tikzLibraries?: string;
    addToPreamble?: string;
    embedFontCss?: boolean;
    fontCssUrl?: string;
    disableOptimize?: boolean;
}
declare const TikzJax: QuartzTransformerPlugin<Partial<TikzJaxOptions>>;

export { TikzJax, type TikzJaxOptions, TikzJax as default };

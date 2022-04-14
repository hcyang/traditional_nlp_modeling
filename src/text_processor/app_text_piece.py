# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests TextPiece objects.
"""

from typing import List

# import os
# import sys

import argparse

# from utility.io_helper.io_helper \
#     import IoHelper

# from utility.datatype_helper.datatype_helper \
#     import DatatypeHelper
from utility.debugging_helper.debugging_helper \
    import DebuggingHelper

from text_processor.text_piece \
    import TextPiece
from text_processor.text_piece \
    import TextPieceListExtractedChineseBookEnclosure
from text_processor.text_piece \
    import TextPieceListExtractedWikiEmphasisBold
from text_processor.text_piece \
    import TextPieceListExtractedWikiEmphasisBoldItalic
from text_processor.text_piece \
    import TextPieceListExtractedWikiLink
# from text_processor.text_piece \
#     import TextPieceListFilteredWikiTemplateArgument
from text_processor.text_piece \
    import TextPieceListFilteredWikiTemplate
from text_processor.text_piece \
    import TextPieceListFilteredHtmlGenericIndividualTag
from text_processor.text_piece \
    import TextPieceListFilteredHtmlReference
from text_processor.text_piece \
    import TextPieceListFilteredComment

def process_text_piece_arguments(parser):
    """
    To process data manager related arguments.
    """
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'Calling process_text_piece_arguments() in {__name__}')
    if parser is None:
        DebuggingHelper.throw_exception(
            'input argument, parser, is None')
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- parser.add_argument(
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     '--rootpath',
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     type=str,
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     required=True,
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     help='Root path for input Wikipedia dump and output processed files.')
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- parser.add_argument(
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     '--filename',
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     type=str,
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     required=True,
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     help='Wikipedia dump input filename.')
    return parser

# ---- NOTE-PYLINT ---- C0103: Function name "" doesn't conform to snake_case naming style
# pylint: disable=C0103
def example_function_TextPiece():
    """
    The main function to quickly test TextPiece.
    """
    # ---- NOTE-PYLINT ---- R0914: Too many local variables
    # pylint: disable=R0914
    # ---- NOTE-PYLINT ---- R0915: Too many statements
    # pylint: disable=R0915
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_text_piece_arguments(parser)
    # ------------------------------------------------------------------------
    # args: argparse.Namespace = parser.parse_args()
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'sys.path={str(sys.path)}')
    # DebuggingHelper.write_line_to_system_console_out(
    #     f'args={str(args)}')
    # ------------------------------------------------------------------------
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- wikipedia_root_process_path: str = \
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     args.rootpath
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- wikipedia_dump_xml_filename: str = \
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     args.filename
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- if DatatypeHelper.is_none_empty_whitespaces_or_nan(wikipedia_root_process_path):
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     error_message: str = \
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----         f'ERROR: no process root path for the "rootpath" argument, args={str(args)}'
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     DebuggingHelper.write_line_to_system_console_err(
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----         error_message)
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     DebuggingHelper.print_in_color(error_message)
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     return
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- if DatatypeHelper.is_none_empty_whitespaces_or_nan(wikipedia_dump_xml_filename):
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     error_message: str = \
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----         f'ERROR: no input for the "filename" argument, args={str(args)}'
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     DebuggingHelper.write_line_to_system_console_err(
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----         error_message)
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     DebuggingHelper.print_in_color(error_message)
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     return
    # ------------------------------------------------------------------------
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- wikipedia_dump_xml_path: str = \
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     os.path.join(wikipedia_root_process_path, wikipedia_dump_xml_filename)
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ---- wikipedia_dump_xml_content: str = \
    # ---- NOTE-PLACE-HOLDER-FOR-REFERENCE ----     IoHelper.read_all_from_file(wikipedia_dump_xml_path)
    # ------------------------------------------------------------------------
    input_text: str = \
        "《'''老子'''》，又名《'''道德經'''》，是[[先秦]]時期的古籍，相傳為[[春秋時期|春秋]]末期思想家[[老子]]所著<ref>阎纯德：《汉学研究》 第8期 第459页 9 《马王堆本〈老子〉及其文献流传的线索》 中华书局, 2004</ref>。《老子》為[[东周|春秋戰國]]時期[[道家]]学派的代表性經典，亦是[[道教]]尊奉的經典。至唐代，[[唐太宗]]命人將《道德經》譯為[[梵語]]；[[唐玄宗]]时，尊此经为《'''道德眞經'''》。"
    # ------------------------------------------------------------------------
    text_piece_list_filtered_comment: TextPieceListFilteredComment = \
        TextPieceListFilteredComment(raw_text_string=input_text)
    text_piece_list_after_filtered_comment: List[TextPiece] = \
        text_piece_list_filtered_comment.get_text_piece_list()
    text_piece_string_after_filtered_comment: str = \
        text_piece_list_filtered_comment.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_filtered_comment)={len(text_piece_list_after_filtered_comment)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_filtered_comment={text_piece_string_after_filtered_comment}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_filtered_comment)={len(text_piece_string_after_filtered_comment)}')
    input_text_without_comment: str = \
        "《'''老子'''》，又名《'''道德經'''》，是[[先秦]]時期的古籍，相傳為[[春秋時期|春秋]]末期思想家[[老子]]所著<ref>阎纯德：《汉学研究》 第8期 第459页 9 《马王堆本〈老子〉及其文献流传的线索》 中华书局, 2004</ref>。《老子》為[[东周|春秋戰國]]時期[[道家]]学派的代表性經典，亦是[[道教]]尊奉的經典。至唐代，[[唐太宗]]命人將《道德經》譯為[[梵語]]；[[唐玄宗]]时，尊此经为《'''道德眞經'''》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_without_comment)={len(input_text_without_comment)}')
    DebuggingHelper.ensure(
        text_piece_string_after_filtered_comment == input_text_without_comment,
        'text_piece_string_after_filtered_comment|{}| != input_text_without_comment|{}|'.format(
            len(text_piece_string_after_filtered_comment),
            len(input_text_without_comment)))
    # ------------------------------------------------------------------------
    text_piece_list_filtered_wiki_template: TextPieceListFilteredWikiTemplate = \
        TextPieceListFilteredWikiTemplate(raw_text_string=text_piece_string_after_filtered_comment)
    text_piece_list_after_filtered_wiki_template: List[TextPiece] = \
        text_piece_list_filtered_wiki_template.get_text_piece_list()
    text_piece_string_after_filtered_wiki_template: str = \
        text_piece_list_filtered_wiki_template.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_filtered_wiki_template)={len(text_piece_list_after_filtered_wiki_template)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_filtered_wiki_template={text_piece_string_after_filtered_wiki_template}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_filtered_wiki_template)={len(text_piece_string_after_filtered_wiki_template)}')
    input_text_without_wiki_template: str = \
        "《'''老子'''》，又名《'''道德經'''》，是[[先秦]]時期的古籍，相傳為[[春秋時期|春秋]]末期思想家[[老子]]所著<ref>阎纯德：《汉学研究》 第8期 第459页 9 《马王堆本〈老子〉及其文献流传的线索》 中华书局, 2004</ref>。《老子》為[[东周|春秋戰國]]時期[[道家]]学派的代表性經典，亦是[[道教]]尊奉的經典。至唐代，[[唐太宗]]命人將《道德經》譯為[[梵語]]；[[唐玄宗]]时，尊此经为《'''道德眞經'''》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_without_wiki_template)={len(input_text_without_wiki_template)}')
    DebuggingHelper.ensure(
        text_piece_string_after_filtered_wiki_template == input_text_without_wiki_template,
        'text_piece_string_after_filtered_wiki_template|{}| != input_text_without_wiki_template|{}|'.format(
            len(text_piece_string_after_filtered_wiki_template),
            len(input_text_without_wiki_template)))
    # ------------------------------------------------------------------------
    text_piece_list_filtered_html_reference: TextPieceListFilteredHtmlReference = \
        TextPieceListFilteredHtmlReference(raw_text_string=text_piece_string_after_filtered_wiki_template)
    text_piece_list_after_filtered_html_reference: List[TextPiece] = \
        text_piece_list_filtered_html_reference.get_text_piece_list()
    text_piece_string_after_filtered_html_reference: str = \
        text_piece_list_filtered_html_reference.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_filtered_html_reference)={len(text_piece_list_after_filtered_html_reference)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_filtered_html_reference={text_piece_string_after_filtered_html_reference}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_filtered_html_reference)={len(text_piece_string_after_filtered_html_reference)}')
    input_text_without_html_reference: str = \
        "《'''老子'''》，又名《'''道德經'''》，是[[先秦]]時期的古籍，相傳為[[春秋時期|春秋]]末期思想家[[老子]]所著。《老子》為[[东周|春秋戰國]]時期[[道家]]学派的代表性經典，亦是[[道教]]尊奉的經典。至唐代，[[唐太宗]]命人將《道德經》譯為[[梵語]]；[[唐玄宗]]时，尊此经为《'''道德眞經'''》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_without_html_reference)={len(input_text_without_html_reference)}')
    DebuggingHelper.ensure(
        text_piece_string_after_filtered_html_reference == input_text_without_html_reference,
        'text_piece_string_after_filtered_html_reference|{}| != input_text_without_html_reference|{}|'.format(
            len(text_piece_string_after_filtered_html_reference),
            len(input_text_without_html_reference)))
    # ------------------------------------------------------------------------
    text_piece_list_filtered_html_generic_individual_tag: TextPieceListFilteredHtmlGenericIndividualTag = \
        TextPieceListFilteredHtmlGenericIndividualTag(raw_text_string=text_piece_string_after_filtered_html_reference)
    text_piece_list_after_filtered_html_generic_individual_tag: List[TextPiece] = \
        text_piece_list_filtered_html_generic_individual_tag.get_text_piece_list()
    text_piece_string_after_filtered_html_generic_individual_tag: str = \
        text_piece_list_filtered_html_generic_individual_tag.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_filtered_html_generic_individual_tag)={len(text_piece_list_after_filtered_html_generic_individual_tag)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_filtered_html_generic_individual_tag={text_piece_string_after_filtered_html_generic_individual_tag}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_filtered_html_generic_individual_tag)={len(text_piece_string_after_filtered_html_generic_individual_tag)}')
    input_text_without_html_generic_individual_tag: str = \
        "《'''老子'''》，又名《'''道德經'''》，是[[先秦]]時期的古籍，相傳為[[春秋時期|春秋]]末期思想家[[老子]]所著。《老子》為[[东周|春秋戰國]]時期[[道家]]学派的代表性經典，亦是[[道教]]尊奉的經典。至唐代，[[唐太宗]]命人將《道德經》譯為[[梵語]]；[[唐玄宗]]时，尊此经为《'''道德眞經'''》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_without_html_generic_individual_tag)={len(input_text_without_html_generic_individual_tag)}')
    DebuggingHelper.ensure(
        text_piece_string_after_filtered_html_generic_individual_tag == input_text_without_html_generic_individual_tag,
        'text_piece_string_after_filtered_html_generic_individual_tag|{}| != input_text_without_html_generic_individual_tag|{}|'.format(
            len(text_piece_string_after_filtered_html_generic_individual_tag),
            len(input_text_without_html_generic_individual_tag)))
    # ------------------------------------------------------------------------
    text_piece_list_extracted_wiki_link: TextPieceListExtractedWikiLink = \
        TextPieceListExtractedWikiLink(raw_text_string=text_piece_string_after_filtered_html_generic_individual_tag)
    text_piece_list_after_extracted_wiki_link: List[TextPiece] = \
        text_piece_list_extracted_wiki_link.get_text_piece_list()
    text_piece_string_after_extracted_wiki_link: str = \
        text_piece_list_extracted_wiki_link.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_extracted_wiki_link)={len(text_piece_list_after_extracted_wiki_link)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_extracted_wiki_link={text_piece_string_after_extracted_wiki_link}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_extracted_wiki_link)={len(text_piece_string_after_extracted_wiki_link)}')
    input_text_with_link_text_piece_string: str = \
        "《'''老子'''》，又名《'''道德經'''》，是先秦時期的古籍，相傳為春秋末期思想家老子所著。《老子》為春秋戰國時期道家学派的代表性經典，亦是道教尊奉的經典。至唐代，唐太宗命人將《道德經》譯為梵語；唐玄宗时，尊此经为《'''道德眞經'''》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_with_link_text_piece_string)={len(input_text_with_link_text_piece_string)}')
    DebuggingHelper.ensure(
        text_piece_string_after_extracted_wiki_link == input_text_with_link_text_piece_string,
        'text_piece_string_after_extracted_wiki_link|{}| != input_text_with_link_text_piece_string|{}|'.format(
            len(text_piece_string_after_extracted_wiki_link),
            len(input_text_with_link_text_piece_string)))
    # ------------------------------------------------------------------------
    text_piece_list_extracted_wiki_emphasis_bold_italic: TextPieceListExtractedWikiEmphasisBoldItalic = \
        TextPieceListExtractedWikiEmphasisBoldItalic(raw_text_string=text_piece_string_after_extracted_wiki_link)
    text_piece_list_after_extracted_wiki_emphasis_bold_italic: List[TextPiece] = \
        text_piece_list_extracted_wiki_emphasis_bold_italic.get_text_piece_list()
    text_piece_string_after_extracted_wiki_emphasis_bold_italic: str = \
        text_piece_list_extracted_wiki_emphasis_bold_italic.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_extracted_wiki_emphasis_bold_italic)={len(text_piece_list_after_extracted_wiki_emphasis_bold_italic)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_extracted_wiki_emphasis_bold_italic={text_piece_string_after_extracted_wiki_emphasis_bold_italic}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_extracted_wiki_emphasis_bold_italic)={len(text_piece_string_after_extracted_wiki_emphasis_bold_italic)}')
    input_text_with_bold_italic_text_piece_string: str = \
        "《'''老子'''》，又名《'''道德經'''》，是先秦時期的古籍，相傳為春秋末期思想家老子所著。《老子》為春秋戰國時期道家学派的代表性經典，亦是道教尊奉的經典。至唐代，唐太宗命人將《道德經》譯為梵語；唐玄宗时，尊此经为《'''道德眞經'''》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_with_bold_italic_text_piece_string)={len(input_text_with_bold_italic_text_piece_string)}')
    DebuggingHelper.ensure(
        text_piece_string_after_extracted_wiki_emphasis_bold_italic == input_text_with_bold_italic_text_piece_string,
        'text_piece_string_after_extracted_wiki_emphasis_bold_italic|{}| != input_text_with_bold_italic_text_piece_string|{}|'.format(
            len(text_piece_string_after_extracted_wiki_emphasis_bold_italic),
            len(input_text_with_bold_italic_text_piece_string)))
    # ------------------------------------------------------------------------
    text_piece_list_extracted_wiki_emphasis_bold: TextPieceListExtractedWikiEmphasisBold = \
        TextPieceListExtractedWikiEmphasisBold(raw_text_string=text_piece_string_after_extracted_wiki_emphasis_bold_italic)
    text_piece_list_after_extracted_wiki_emphasis_bold: List[TextPiece] = \
        text_piece_list_extracted_wiki_emphasis_bold.get_text_piece_list()
    text_piece_string_after_extracted_wiki_emphasis_bold: str = \
        text_piece_list_extracted_wiki_emphasis_bold.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_extracted_wiki_emphasis_bold)={len(text_piece_list_after_extracted_wiki_emphasis_bold)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_extracted_wiki_emphasis_bold={text_piece_string_after_extracted_wiki_emphasis_bold}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_extracted_wiki_emphasis_bold)={len(text_piece_string_after_extracted_wiki_emphasis_bold)}')
    input_text_with_bold_text_piece_string: str = \
        "《老子》，又名《道德經》，是先秦時期的古籍，相傳為春秋末期思想家老子所著。《老子》為春秋戰國時期道家学派的代表性經典，亦是道教尊奉的經典。至唐代，唐太宗命人將《道德經》譯為梵語；唐玄宗时，尊此经为《道德眞經》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_with_bold_text_piece_string)={len(input_text_with_bold_text_piece_string)}')
    DebuggingHelper.ensure(
        text_piece_string_after_extracted_wiki_emphasis_bold == input_text_with_bold_text_piece_string,
        'text_piece_string_after_extracted_wiki_emphasis_bold|{}| != input_text_with_bold_text_piece_string|{}|'.format(
            len(text_piece_string_after_extracted_wiki_emphasis_bold),
            len(input_text_with_bold_text_piece_string)))
    # ------------------------------------------------------------------------
    text_piece_list_extracted_chinese_book_enclosure: TextPieceListExtractedChineseBookEnclosure = \
        TextPieceListExtractedChineseBookEnclosure(raw_text_string=text_piece_string_after_extracted_wiki_emphasis_bold)
    text_piece_list_after_extracted_chinese_book_enclosure: List[TextPiece] = \
        text_piece_list_extracted_chinese_book_enclosure.get_text_piece_list()
    text_piece_string_after_extracted_chinese_book_enclosure: str = \
        text_piece_list_extracted_chinese_book_enclosure.get_text_piece_string()
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_list_after_extracted_chinese_book_enclosure)={len(text_piece_list_after_extracted_chinese_book_enclosure)}')
    # DebuggingHelper.write_line_to_system_console_out_debug(
    #     message=f'text_piece_string_after_extracted_chinese_book_enclosure={text_piece_string_after_extracted_chinese_book_enclosure}')
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(text_piece_string_after_extracted_chinese_book_enclosure)={len(text_piece_string_after_extracted_chinese_book_enclosure)}')
    input_text_with_bold_text_piece_string: str = \
        "《老子》，又名《道德經》，是先秦時期的古籍，相傳為春秋末期思想家老子所著。《老子》為春秋戰國時期道家学派的代表性經典，亦是道教尊奉的經典。至唐代，唐太宗命人將《道德經》譯為梵語；唐玄宗时，尊此经为《道德眞經》。"
    DebuggingHelper.write_line_to_system_console_out_debug(
        message=f'len(input_text_with_bold_text_piece_string)={len(input_text_with_bold_text_piece_string)}')
    DebuggingHelper.ensure(
        text_piece_string_after_extracted_chinese_book_enclosure == input_text_with_bold_text_piece_string,
        'text_piece_string_after_extracted_chinese_book_enclosure|{}| != input_text_with_bold_text_piece_string|{}|'.format(
            len(text_piece_string_after_extracted_chinese_book_enclosure),
            len(input_text_with_bold_text_piece_string)))
    # ------------------------------------------------------------------------

def main():
    """
    The main() function can quickly test TextPiece objects.
    """
    example_function_TextPiece()

if __name__ == '__main__':
    main()
    # ---- NOTE-PYLINT ---- C0301: Line too long
    # pylint: disable=C0301
    # ---- NOTE-FOR-REFERENCE ---- python app_text_piece.py
    # ---- NOTE-FOR-REFERENCE ---- python app_text_piece.py --rootpath E:\data_wikipedia\dumps.wikimedia.org_zhwiki_20211020 --filename zhwiki-20211020-pages-articles.xml

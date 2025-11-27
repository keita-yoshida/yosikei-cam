def dxf_to_shapely_polygon(uploaded_file):
    """DXFファイルを読み込み、Shapely Polygonに変換する (PLINE, LWPOLYLINE, LINEのみ対応)"""
    
    if uploaded_file is None:
        return None, "ファイルがアップロードされていません。"
    
    try:
        # 1. ファイルポインタを先頭に戻す
        uploaded_file.seek(0)
        
        # 2. ファイルの内容をバイト列として取得
        # .getvalue() は通常bytesを返すが、万が一strが返った場合も対応するため変数に格納
        file_content = uploaded_file.getvalue()
        
        # 3. 強制的なバイト列チェックと変換 (エラー回避策)
        if isinstance(file_content, str):
            # もしファイル内容がstr型として読み込まれていたら、バイト列にエンコード
            dxf_bytes = file_content.encode('utf-8') 
        else:
            # bytes型ならそのまま使用
            dxf_bytes = file_content
            
        # 4. BytesIOに格納し、ezdxfに渡す
        doc = ezdxf.read(BytesIO(dxf_bytes))
        msp = doc.modelspace()
        
        polylines = []
        
        for entity in msp:
            # 構文エラーがないことを確認
            if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
                coords = [(p[0], p[1]) for p in entity.vertices()]
                
                if entity.is_closed:
                    try:
                        polylines.append(Polygon(coords))
                    except Exception:
                        pass
                else:
                    pass 
        
        if not polylines:
            return None, "DXFファイル内に閉じたポリライン (LWPOLYLINE/POLYLINE) が見つかりませんでした。"
        
        if len(polylines) > 1:
            main_polygon = max(polylines, key=lambda p: p.area)
            return main_polygon, f"複数の図形を検出。最大面積の図形（頂点数: {len(main_polygon.exterior.coords)}）を採用しました。"
        else:
            return polylines[0], f"図形を検出しました。（頂点数: {len(polylines[0].exterior.coords)}）"

    except ezdxf.DXFStructureError as e:
        return None, f"DXFファイルの構造エラーです: {e}"
    except Exception as e:
        return None, f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}"
